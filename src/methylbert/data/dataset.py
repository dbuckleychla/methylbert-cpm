import bz2
import gc
import gzip
import hashlib
import json
import lzma
import multiprocessing as mp
import os
import random
import time
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from methylbert.data.vocab import MethylVocab
try:
	from tqdm import tqdm
except Exception:
	tqdm = None


def _open_text(path: str):
	if path.endswith(".gz"):
		return gzip.open(path, "rt")
	if path.endswith(".bz2"):
		return bz2.open(path, "rt")
	if path.endswith(".xz"):
		return lzma.open(path, "rt")
	return open(path, "r")

def _is_parquet(path: str) -> bool:
	path = path.lower()
	return path.endswith(".parquet") or path.endswith(".pq")

def _torchrun_rank():
	rank = os.environ.get("RANK")
	world_size = os.environ.get("WORLD_SIZE")
	if rank is None or world_size is None:
		return None
	try:
		return int(rank)
	except ValueError:
		return None

def _token_cache_prefix(f_path: str, seq_len: int, kmer: int, cache_dir: str, n_seqs: int = None) -> str:
	abs_path = os.path.abspath(f_path)
	hash_id = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:10]
	base = f"{os.path.basename(f_path)}.{hash_id}.seq{seq_len}.k{kmer}"
	if n_seqs is not None:
		base = f"{base}.n{n_seqs}"
	return os.path.join(cache_dir, base)

def _line2tokens_pretrain(l, tokenizer, max_len=120):
	'''
		convert a text line into a list of tokens converted by tokenizer

	'''

	l = l.strip().split(" ")

	tokened = [tokenizer.to_seq(b) for b in l]
	if len(tokened) > max_len:
		return tokened[:max_len]
	else:
		return tokened + [[tokenizer.pad_index] for k in range(max_len-len(tokened))]

def _parse_line(l, headers):
	# Check the header
	if not all([h in headers for h in ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"]]):
		raise ValueError("The header must contain dna_seq, methyl_seq, ctype, dmr_ctype, dmr_label")

	# Separate n-mers tokens and labels from each line
	l = l.split("\t")  # don't add strip; some columns may be None
	if len(headers) == len(l):
		l = {k: v for k, v in zip(headers, l)}
	else:
		raise ValueError(f"Only {len(headers)} elements are in the input file header, whereas the line has {len(l)} elements.")

	# Cell-type label is binary (whether the cell type corresponds to the DMR cell type)
	l["ctype_label"] = int(l["ctype"] == l["dmr_ctype"])
	l["dmr_label"] = int(l["dmr_label"])

	return l


def _line2tokens_finetune(l, tokenizer, max_len=150, headers=None):
	# parsed line!

	l["dna_seq"] = l["dna_seq"].split(" ")
	l["dna_seq"] = [[f] for f in tokenizer.to_seq(l["dna_seq"])]
	l["methyl_seq"] = [int(m) for m in l["methyl_seq"]]

	if len(l["dna_seq"]) > max_len:
		l["dna_seq"] = l["dna_seq"][:max_len]
		l["methyl_seq"] = l["methyl_seq"][:max_len]
	else:
		cur_seq_len=len(l["dna_seq"])
		l["dna_seq"] = l["dna_seq"]+[[tokenizer.pad_index] for k in range(max_len-cur_seq_len)]
		l["methyl_seq"] = l["methyl_seq"] + [2 for k in range(max_len-cur_seq_len)]

	return l

_TOKEN_CACHE_WORKER_CTX = {}

def _init_token_cache_worker(vocab, seq_len, headers):
	global _TOKEN_CACHE_WORKER_CTX
	_TOKEN_CACHE_WORKER_CTX = {
		"vocab": vocab,
		"seq_len": seq_len,
		"headers": headers,
	}

def _tokenize_cache_record(args):
	idx, rec = args
	ctx = _TOKEN_CACHE_WORKER_CTX
	tokenizer = ctx["vocab"]
	seq_len = ctx["seq_len"]
	headers = ctx["headers"]

	if isinstance(rec, str):
		parsed = _parse_line(rec, headers)
	else:
		parsed = rec
		parsed["ctype_label"] = int(parsed["ctype"] == parsed["dmr_ctype"])
		parsed["dmr_label"] = int(parsed["dmr_label"])

	item = _line2tokens_finetune(
		l=parsed,
		tokenizer=tokenizer, max_len=seq_len, headers=headers)

	dna_seq = np.array(item["dna_seq"], dtype=np.int32).squeeze()
	methyl_seq = np.array(item["methyl_seq"], dtype=np.int8).squeeze()

	non_pad = np.where(dna_seq != tokenizer.pad_index)[0]
	if non_pad.size > 0:
		end = non_pad[-1] + 1
		if end < dna_seq.shape[0]:
			dna_seq[end] = tokenizer.eos_index
			methyl_seq[end] = 2
		else:
			dna_seq[-1] = tokenizer.eos_index
			methyl_seq[-1] = 2

	dna_seq = np.concatenate([[tokenizer.sos_index], dna_seq])
	methyl_seq = np.concatenate([[2], methyl_seq])

	return idx, dna_seq, methyl_seq, parsed["dmr_label"], parsed["ctype_label"]

class MethylBertDataset(Dataset):
	def __init__(self):
		pass

	def __len__(self):
		return self.lines.shape[0] if type(self.lines) == np.array else len(self.lines)


class MethylBertPretrainDataset(MethylBertDataset):
	def __init__(self, f_path: str, vocab: MethylVocab, seq_len: int, random_len=False, n_cores=50):

		self.vocab = vocab
		self.seq_len = seq_len
		self.f_path = f_path
		self.random_len = random_len

		# Define a range of tokens to mask based on k-mers
		self.mask_list = self._get_mask()

		# Read all text files and convert the raw sequence into tokens
		with _open_text(self.f_path) as f_input:
			print("Open data : %s"%f_input)
			raw_seqs = f_input.read().splitlines()

		print("Total number of sequences : ", len(raw_seqs))

		# Multiprocessing for the sequence tokenisation
		with mp.Pool(n_cores) as pool:
			line_labels = pool.map(partial(_line2tokens_pretrain,
								           tokenizer=self.vocab,
								           max_len=self.seq_len), raw_seqs)
			del raw_seqs
			print("Lines are processed")
			self.lines = torch.squeeze(torch.tensor(np.array(line_labels, dtype=np.int16)))
		del line_labels
		gc.collect()

	def __getitem__(self, index):

		dna_seq = self.lines[index].clone()

		# Random len
		if self.random_len and np.random.random() < 0.5:
			dna_seq = dna_seq[:random.randint(5, self.seq_len)]

		# Padding
		if dna_seq.shape[0] < self.seq_len:
			pad_num = self.seq_len-dna_seq.shape[0]
			dna_seq = torch.cat((dna_seq,
								torch.tensor([self.vocab.pad_index for i in range(pad_num)], dtype=torch.int16)))

		# Mask
		masked_dna_seq, dna_seq, bert_mask = self._masking(dna_seq)
		#print(dna_seq, masked_dna_seq,"\n=============================================\n")
		return {"bert_input": masked_dna_seq,
				"bert_label": dna_seq,
				"bert_mask" : bert_mask}

	def subset_data(self, n_seq: int):
		self.lines = random.sample(self.lines, n_seq)

	def _get_mask(self):
		'''
			Relative positions from the centre of masked region
			e.g) [-1, 0, 1] for 3-mers
		'''
		half_length = int(self.vocab.kmers/2)
		mask_list = [-1*half_length + i for i in range(half_length)] + [i for i in range(1, half_length+1)]
		if self.vocab.kmers % 2 == 0:
			mask_list = mask_list[:-1]

		return mask_list

	def _masking(self, inputs: torch.Tensor, threshold=0.15):
		"""
			Moidfied version of masking token function
			Originally developed by Huggingface (datacollator) and DNABERT

			https://github.com/huggingface/transformers/blob/9a24b97b7f304fa1ceaaeba031241293921b69d3/src/transformers/data/data_collator.py#L747

			https://github.com/jerryji1993/DNABERT/blob/bed72fc0694a7b04f7e980dc9ce986e2bb785090/examples/run_pretrain.py#L251

			Added additional tasks to handle each sequence
			Lines using tokenizer were modified due to different tokenizer object structure

		"""

		labels = inputs.clone()

		# Sample tokens with given probability threshold
		probability_matrix = torch.full(labels.shape, threshold) # tensor filled with 0.15

		# Handle special tokens and padding
		special_tokens_mask = [
			val < 5 for val in labels.tolist()
		]
		probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
		#padding_mask = labels.eq(self.vocab.pad_index)
		#probability_matrix.masked_fill_(padding_mask, value=0.0)

		masked_indices = torch.bernoulli(probability_matrix).bool() # get masked tokens based on bernoulli only within non-special tokens

		# change masked indices
		masked_index = deepcopy(masked_indices)

		# This function handles each sequence
		end = torch.where(probability_matrix!=0)[0].tolist()[-1] # end of the sequence
		mask_centers = set(torch.where(masked_index==1)[0].tolist()) # mask locations

		new_centers = deepcopy(mask_centers)
		for center in mask_centers:
			for mask_number in self.mask_list:# add neighbour loci
				current_index = center + mask_number
				if current_index <= end and current_index >= 0:
					new_centers.add(current_index)

		new_centers = list(new_centers)

		masked_indices[new_centers] = True

		# Avoid loss calculation on unmasked tokens
		labels[~masked_indices] = -100

		# 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
		indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
		inputs[indices_replaced] = self.vocab.mask_index

		# 10% of the time, we replace masked input tokens with random word
		indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
		random_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.int16)
		inputs[indices_random] = random_words[indices_random]

		# The rest of the time (10% of the time) we keep the masked input tokens unchanged

		# Special tokens (SOS, EOS)
		if end < inputs.shape[0]:
			inputs[end] = self.vocab.eos_index
		else:
			inputs[-1] = self.vocab.eos_index

		labels = torch.cat((torch.tensor([-100]), labels))
		inputs = torch.cat((torch.tensor([self.vocab.sos_index]), inputs))
		masked_index = torch.cat((torch.tensor([False]), masked_index))


		return inputs, labels, masked_index

class MethylBertFinetuneDataset(MethylBertDataset):
	def __init__(self, f_path: str, vocab: MethylVocab, seq_len: int, n_cores: int=10, n_seqs = None,
				 token_cache_dir: str = None, rebuild_token_cache: bool = False,
				 token_cache_timeout_s: int = 3600, token_cache_workers: int = None,
				 token_cache_progress: bool = False, token_cache_mode: str = "auto"):
		'''
		MethylBERT dataset

		f_path: str
			File path to the processed input file
		vocab: MethylVocab
			MethylVocab object to convert DNA and methylation pattern sequences
		seq_len: int
			Length for the processed sequences
		n_cores: int
			Number of cores for multiprocessing
		n_seqs: int
			Number of sequences to subset the input (default: None, do not make a subset)
		token_cache_dir: str
			Directory to store/load pre-tokenized arrays (default: None)
		rebuild_token_cache: bool
			Rebuild token cache even if present (default: False)
		token_cache_timeout_s: int
			Seconds to wait for token cache (default: 3600). Set to 0 or negative to wait indefinitely.
		token_cache_workers: int
			Workers to use for token cache build (default: all CPUs).
		token_cache_progress: bool
			Show a progress bar during token cache build (default: False).
		token_cache_mode: str
			Cache behavior: auto (build if missing), build (force build), load (load only).

		'''
		self.vocab = vocab
		self.seq_len = seq_len
		self.f_path = f_path
		self._use_token_cache = False
		self._length = None
		self._cache_paths = None
		self._token_cache_workers = token_cache_workers
		if self._token_cache_workers is not None and self._token_cache_workers <= 0:
			self._token_cache_workers = None
		self._token_cache_progress = token_cache_progress
		self._token_cache_mode = token_cache_mode or "auto"

		if token_cache_timeout_s is not None and token_cache_timeout_s <= 0:
			token_cache_timeout_s = None

		if token_cache_dir is not None:
			os.makedirs(token_cache_dir, exist_ok=True)
			prefix = _token_cache_prefix(
				self.f_path,
				self.seq_len,
				self.vocab.kmers,
				token_cache_dir,
				n_seqs=n_seqs,
			)
			self._cache_paths = {
				"dna": f"{prefix}.dna.npy",
				"methyl": f"{prefix}.methyl.npy",
				"dmr_label": f"{prefix}.dmr_label.npy",
				"ctype_label": f"{prefix}.ctype_label.npy",
				"meta": f"{prefix}.meta.json",
				"lock": f"{prefix}.lock",
			}
			cache_files = ("dna", "methyl", "dmr_label", "ctype_label", "meta")
			cache_ready = all(os.path.exists(self._cache_paths[k]) for k in cache_files)
			torchrun_rank = _torchrun_rank()
			can_build_cache = torchrun_rank is None or torchrun_rank == 0
			if self._token_cache_mode == "load":
				if not cache_ready:
					missing = [k for k in cache_files if not os.path.exists(self._cache_paths[k])]
					raise FileNotFoundError(
						f"Token cache missing: {missing}. Run build_token_cache first."
					)
				self._load_token_cache()
			elif self._token_cache_mode == "build":
				if can_build_cache:
					built = self._build_token_cache(n_seqs=n_seqs)
					if not built:
						self._wait_for_token_cache(
							timeout_s=token_cache_timeout_s,
							wait_for_refresh=rebuild_token_cache,
						)
				else:
					self._wait_for_token_cache(
						timeout_s=token_cache_timeout_s,
						wait_for_refresh=rebuild_token_cache,
					)
				self._load_token_cache()
			else:
				if cache_ready and not rebuild_token_cache:
					self._load_token_cache()
				else:
					if can_build_cache:
						built = self._build_token_cache(n_seqs=n_seqs)
						if not built:
							self._wait_for_token_cache(
								timeout_s=token_cache_timeout_s,
								wait_for_refresh=rebuild_token_cache,
							)
					else:
						self._wait_for_token_cache(
							timeout_s=token_cache_timeout_s,
							wait_for_refresh=rebuild_token_cache,
						)
					self._load_token_cache()

		if self._use_token_cache:
			return

		if _is_parquet(self.f_path):
			df_reads = pd.read_parquet(self.f_path)
			self.headers = df_reads.columns.tolist()
			required_headers = ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"]
			if not all([h in self.headers for h in required_headers]):
				raise ValueError("The header must contain dna_seq, methyl_seq, ctype, dmr_ctype, dmr_label")

			if n_seqs is not None:
				df_reads = df_reads.head(n_seqs)
			print("Total number of sequences : ", df_reads.shape[0])

			self.lines = []
			for row in df_reads.to_dict("records"):
				row["ctype_label"] = int(row["ctype"] == row["dmr_ctype"])
				row["dmr_label"] = int(row["dmr_label"])
				self.lines.append(row)
			del df_reads
			gc.collect()
		else:
			# Read all text files and convert the raw sequence into tokens
			with _open_text(self.f_path) as f_input:
				raw_seqs = f_input.read().splitlines()

			# Check if there's a header
			self.headers = raw_seqs[0].split("\t")
			raw_seqs = raw_seqs[1:]

			if n_seqs is not None:
				raw_seqs = raw_seqs[:n_seqs]
			print("Total number of sequences : ", len(raw_seqs))

			# Multiprocessing for the sequence tokenisation
			with mp.Pool(n_cores) as pool:
				self.lines = pool.map(partial(_parse_line,
									   headers=self.headers), raw_seqs)
				del raw_seqs
			gc.collect()
		self.set_dmr_labels = set([l["dmr_label"] for l in self.lines])

		self.ctype_label_count = self._get_cls_num()
		print("# of reads in each label: ", self.ctype_label_count)

	def _build_token_cache(self, n_seqs=None):
		lock_path = self._cache_paths.get("lock") if self._cache_paths else None
		lock_acquired = False
		progress = None
		progress_update_every = None
		progress_count = 0
		if lock_path is not None:
			try:
				fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
				with os.fdopen(fd, "w") as fp:
					fp.write(f"{os.getpid()}\n")
				lock_acquired = True
			except FileExistsError:
				return False
		else:
			lock_acquired = True
		try:
			if _is_parquet(self.f_path):
				df_reads = pd.read_parquet(self.f_path)
				self.headers = df_reads.columns.tolist()
				required_headers = ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"]
				if not all([h in self.headers for h in required_headers]):
					raise ValueError("The header must contain dna_seq, methyl_seq, ctype, dmr_ctype, dmr_label")
				if n_seqs is not None:
					df_reads = df_reads.head(n_seqs)
				records = df_reads.to_dict("records")
				del df_reads
			else:
				with _open_text(self.f_path) as f_input:
					raw_seqs = f_input.read().splitlines()
				self.headers = raw_seqs[0].split("\t")
				raw_seqs = raw_seqs[1:]
				if n_seqs is not None:
					raw_seqs = raw_seqs[:n_seqs]
				records = raw_seqs

			n_rows = len(records)
			print("Pre-tokenizing sequences : ", n_rows)

			if self._token_cache_progress and tqdm is not None and n_rows:
				progress = tqdm(total=n_rows, unit="seq", mininterval=1.0, smoothing=0.05)
				progress_update_every = max(1, min(10000, n_rows // 1000 or 1))

			dna_mm = np.lib.format.open_memmap(self._cache_paths["dna"], mode="w+", dtype=np.int32, shape=(n_rows, self.seq_len + 1))
			methyl_mm = np.lib.format.open_memmap(self._cache_paths["methyl"], mode="w+", dtype=np.int8, shape=(n_rows, self.seq_len + 1))
			dmr_label_mm = np.lib.format.open_memmap(self._cache_paths["dmr_label"], mode="w+", dtype=np.int32, shape=(n_rows,))
			ctype_label_mm = np.lib.format.open_memmap(self._cache_paths["ctype_label"], mode="w+", dtype=np.int8, shape=(n_rows,))

			token_cache_workers = self._token_cache_workers
			if token_cache_workers is None or token_cache_workers <= 0:
				token_cache_workers = os.cpu_count() or 1
			token_cache_workers = min(token_cache_workers, n_rows) if n_rows else 1

			if token_cache_workers > 1:
				chunk_size = max(1, min(1000, n_rows // (token_cache_workers * 4) or 1))
				with mp.Pool(
					token_cache_workers,
					initializer=_init_token_cache_worker,
					initargs=(self.vocab, self.seq_len, self.headers),
				) as pool:
					for idx, dna_seq, methyl_seq, dmr_label, ctype_label in pool.imap_unordered(
						_tokenize_cache_record,
						enumerate(records),
						chunksize=chunk_size,
					):
						dna_mm[idx] = dna_seq
						methyl_mm[idx] = methyl_seq
						dmr_label_mm[idx] = dmr_label
						ctype_label_mm[idx] = ctype_label
						if progress is not None:
							progress_count += 1
							if progress_count % progress_update_every == 0:
								progress.update(progress_update_every)
			else:
				for idx, rec in enumerate(records):
					if isinstance(rec, str):
						parsed = _parse_line(rec, self.headers)
					else:
						parsed = rec
						parsed["ctype_label"] = int(parsed["ctype"] == parsed["dmr_ctype"])
						parsed["dmr_label"] = int(parsed["dmr_label"])

					item = _line2tokens_finetune(
						l=parsed,
						tokenizer=self.vocab, max_len=self.seq_len, headers=self.headers)

					dna_seq = np.array(item["dna_seq"], dtype=np.int32).squeeze()
					methyl_seq = np.array(item["methyl_seq"], dtype=np.int8).squeeze()

					non_pad = np.where(dna_seq != self.vocab.pad_index)[0]
					if non_pad.size > 0:
						end = non_pad[-1] + 1
						if end < dna_seq.shape[0]:
							dna_seq[end] = self.vocab.eos_index
							methyl_seq[end] = 2
						else:
							dna_seq[-1] = self.vocab.eos_index
							methyl_seq[-1] = 2

					dna_seq = np.concatenate([[self.vocab.sos_index], dna_seq])
					methyl_seq = np.concatenate([[2], methyl_seq])

					dna_mm[idx] = dna_seq
					methyl_mm[idx] = methyl_seq
					dmr_label_mm[idx] = parsed["dmr_label"]
					ctype_label_mm[idx] = parsed["ctype_label"]
					if progress is not None:
						progress_count += 1
						if progress_count % progress_update_every == 0:
							progress.update(progress_update_every)

			meta = {
				"seq_len": self.seq_len,
				"source": os.path.abspath(self.f_path),
				"rows": n_rows,
				"kmer": self.vocab.kmers,
				"version": 1,
			}
			with open(self._cache_paths["meta"], "w") as fp:
				json.dump(meta, fp)

			del dna_mm, methyl_mm, dmr_label_mm, ctype_label_mm
			gc.collect()
		finally:
			if progress is not None:
				remainder = progress_count % progress_update_every if progress_update_every else 0
				if remainder:
					progress.update(remainder)
				progress.close()
			if lock_path is not None and lock_acquired and os.path.exists(lock_path):
				os.remove(lock_path)
		return True

	def _wait_for_token_cache(self, timeout_s: int = 3600, poll_s: int = 5, wait_for_refresh: bool = False):
		if not self._cache_paths:
			return
		cache_files = ("dna", "methyl", "dmr_label", "ctype_label", "meta")
		lock_path = self._cache_paths.get("lock")
		meta_path = self._cache_paths.get("meta")
		initial_mtime = os.path.getmtime(meta_path) if meta_path and os.path.exists(meta_path) else None
		saw_missing = False
		start = time.time()
		wait_logged = False
		while True:
			if wait_for_refresh and meta_path and not os.path.exists(meta_path):
				saw_missing = True
			cache_ready = all(os.path.exists(self._cache_paths[k]) for k in cache_files)
			lock_exists = lock_path is not None and os.path.exists(lock_path)
			if cache_ready and not lock_exists:
				if wait_for_refresh and initial_mtime is not None and meta_path and os.path.exists(meta_path):
					if os.path.getmtime(meta_path) > initial_mtime or saw_missing:
						return
				else:
					return
			if not wait_logged:
				print(f"Waiting for token cache build: {self._cache_paths['meta']}")
				wait_logged = True
			if timeout_s is not None and (time.time() - start) > timeout_s:
				missing = [k for k in cache_files if not os.path.exists(self._cache_paths[k])]
				raise TimeoutError(f"Timed out waiting for token cache. Missing: {missing}")
			time.sleep(poll_s)

	def _load_token_cache(self):
		with open(self._cache_paths["meta"], "r") as fp:
			meta = json.load(fp)
		if int(meta.get("seq_len", -1)) != self.seq_len:
			raise ValueError("Token cache seq_len does not match dataset seq_len.")
		meta_kmer = int(meta.get("kmer", self.vocab.kmers))
		if meta_kmer != self.vocab.kmers:
			raise ValueError("Token cache kmer does not match dataset kmer.")

		self._dna_cache = np.load(self._cache_paths["dna"], mmap_mode="r")
		self._methyl_cache = np.load(self._cache_paths["methyl"], mmap_mode="r")
		self._dmr_label_cache = np.load(self._cache_paths["dmr_label"], mmap_mode="r")
		self._ctype_label_cache = np.load(self._cache_paths["ctype_label"], mmap_mode="r")

		self._length = self._dna_cache.shape[0]
		self._use_token_cache = True
		self.headers = ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"]

		self.set_dmr_labels = set(np.unique(self._dmr_label_cache))
		self.ctype_label_count = self._get_cls_num()
		print("# of reads in each label: ", self.ctype_label_count)

	def _get_cls_num(self):
		# unique labels
		if self._use_token_cache:
			ctype_labels = self._ctype_label_cache[:self._length]
		else:
			ctype_labels=[l["ctype_label"] for l in self.lines]
		labels = list(set(ctype_labels))
		label_count = np.zeros(len(labels))
		for l in labels:
			label_count[l] = sum(np.array(ctype_labels) == l)
		return label_count

	def num_dmrs(self):
		return max(len(self.set_dmr_labels), max(self.set_dmr_labels)+1) # +1 is for the label 0

	def subset_data(self, n_seq):
		if self._use_token_cache:
			self._length = min(n_seq, self._length)
		else:
			self.lines = self.lines[:n_seq]

	def __getitem__(self, index):
		if self._use_token_cache:
			return {
				"dna_seq": torch.tensor(self._dna_cache[index], dtype=torch.int32),
				"methyl_seq": torch.tensor(self._methyl_cache[index], dtype=torch.int8),
				"dmr_label": torch.tensor(self._dmr_label_cache[index], dtype=torch.int64),
				"ctype_label": torch.tensor(self._ctype_label_cache[index], dtype=torch.int64),
			}

		line = deepcopy(self.lines[index])
		read_name = line.get("read_name", line.get("name"))

		item = _line2tokens_finetune(
			l=line,
			tokenizer=self.vocab, max_len=self.seq_len, headers=self.headers)

		item["dna_seq"] = torch.squeeze(torch.tensor(np.array(item["dna_seq"], dtype=np.int32)))
		item["methyl_seq"] = torch.squeeze(torch.tensor(np.array(item["methyl_seq"], dtype=np.int8)))

		# Special tokens (SOS, EOS)
		end = torch.where(item["dna_seq"]!=self.vocab.pad_index)[0].tolist()[-1] + 1 # end of the read
		if end < item["dna_seq"].shape[0]:
			item["dna_seq"][end] = self.vocab.eos_index
			item["methyl_seq"][end] = 2
		else:
			item["dna_seq"][-1] = self.vocab.eos_index
			item["methyl_seq"][-1] = 2
		item["dna_seq"] = torch.cat((torch.tensor([self.vocab.sos_index]), item["dna_seq"]))
		item["methyl_seq"] = torch.cat((torch.tensor([2]), item["methyl_seq"]))
		if read_name is not None:
			item["read_name"] = read_name

		return item

	def __len__(self):
		if self._use_token_cache:
			return self._length
		return super().__len__()
