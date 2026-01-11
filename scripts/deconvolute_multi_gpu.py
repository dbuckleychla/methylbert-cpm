#!/usr/bin/env python3
"""
Run MethylBERT deconvolution on one or many BAMs, one BAM per GPU.
Preprocesses each BAM into a temporary bulk dataset before deconvolution.
"""
import argparse
import glob
import hashlib
import os
import shutil
import sys
import tempfile
from multiprocessing import get_context
from pathlib import Path


def _parse_args():
	parser = argparse.ArgumentParser(description="Multi-GPU MethylBERT deconvolution wrapper")
	parser.add_argument("--input_bam_dir", type=str, default=None,
						help="Directory containing BAM files to process")
	parser.add_argument("--bam_file", type=str, default=None,
						help="Single BAM file to process")
	parser.add_argument("--model_dir", type=str, required=True,
						help="Trained MethylBERT model directory (must contain train_param.txt)")
	parser.add_argument("--f_dmr", type=str, required=True, help="DMR .bed/.csv path")
	parser.add_argument("--f_ref", type=str, required=True, help="Reference FASTA path")
	parser.add_argument("--output_dir", type=str, required=True,
						help="Directory to write one output file per BAM")
	parser.add_argument("--tmp_root", type=str, default="methylbert_deconv_tmp",
						help="Base directory for temporary bulk preprocessing outputs")
	parser.add_argument("--gpus", type=str, default="all",
						help="Comma-separated GPU ids to use, or 'all' (default: all)")
	parser.add_argument("--n_mers", type=int, default=3, help="k-mer size (default: 3)")
	parser.add_argument("--n_cores_per_gpu", type=int, default=8,
						help="CPU cores to use per GPU for preprocessing (default: 8)")
	parser.add_argument("--methylcaller", type=str, default="bismark",
						help="Methylation caller: bismark or dorado (default: bismark)")
	parser.add_argument("--output_format", type=str, default="parquet",
						choices=["csv", "parquet"], help="Bulk output format (default: parquet)")
	parser.add_argument("--output_compression", type=str, default="snappy",
						help="Bulk output compression (default: snappy)")
	parser.add_argument("--save_mode", type=str, default="minimal",
						choices=["full", "minimal"], help="Bulk output column mode (default: minimal)")
	parser.add_argument("--batch_size", type=int, default=512,
						help="Deconvolute batch size (default: 64)")
	parser.add_argument("--save_logit", action="store_true",
						help="Save logits from the model (default: False)")
	parser.add_argument("--adjustment", action="store_true",
						help="Adjust estimated tumour purity (default: False)")
	parser.add_argument("--seed", type=int, default=950410, help="Random seed (default: 950410)")
	parser.add_argument("--keep_tmp", action="store_true",
						help="Keep temporary bulk preprocessing outputs")
	parser.add_argument("--keep_all_outputs", action="store_true",
						help="Keep full per-BAM deconvolution output directory")
	return parser.parse_args()


def _parse_gpu_list(gpu_arg: str):
	if gpu_arg and gpu_arg.lower() != "all":
		return [g.strip() for g in gpu_arg.replace(";", ",").split(",") if g.strip()]

	env_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
	if env_visible:
		visible = [g.strip() for g in env_visible.split(",") if g.strip()]
		if visible:
			return visible

	try:
		import torch
		count = torch.cuda.device_count()
		return [str(i) for i in range(count)]
	except Exception:
		return []


def _expected_data_path(output_dir: Path, output_format: str, output_compression: str):
	if output_format == "parquet":
		return output_dir / "data.parquet"

	ext = ".csv"
	compression_suffix = {
		"gzip": ".gz",
		"bz2": ".bz2",
		"xz": ".xz",
	}.get((output_compression or "").lower(), "")
	return output_dir / f"data{ext}{compression_suffix}"


def _sanitize_model_dir(model_dir: str) -> str:
	return model_dir if model_dir.endswith(os.sep) else model_dir + os.sep


def _collect_bams(input_bam_dir: str, bam_file: str):
	bams = []
	if bam_file:
		bams.append(bam_file)
	if input_bam_dir:
		bams.extend(sorted(glob.glob(os.path.join(input_bam_dir, "*.bam"))))
	return bams


def _build_output_name_map(bams):
	name_counts = {}
	for bam in bams:
		name_counts[Path(bam).stem] = name_counts.get(Path(bam).stem, 0) + 1

	name_map = {}
	for bam in bams:
		stem = Path(bam).stem
		if name_counts[stem] > 1:
			hash_id = hashlib.md5(bam.encode("utf-8")).hexdigest()[:8]
			stem = f"{stem}_{hash_id}"
		name_map[bam] = stem
	return name_map


def _worker(gpu_id, job_queue, result_queue, cfg):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

	from argparse import Namespace
	from methylbert.data import finetune_data_generate as fdg
	from methylbert.cli import run_deconvolute

	while True:
		job = job_queue.get()
		if job is None:
			break
		bam_path, output_stem = job
		try:
			tmp_root = Path(cfg["tmp_root"])
			tmp_root.mkdir(parents=True, exist_ok=True)
			tmp_dir = Path(tempfile.mkdtemp(prefix=f"{output_stem}_", dir=tmp_root))

			print(f"[gpu {gpu_id}] preprocessing {bam_path} -> {tmp_dir}", flush=True)
			fdg.finetune_data_generate(
				input_file=bam_path,
				f_dmr=cfg["f_dmr"],
				f_ref=cfg["f_ref"],
				output_dir=str(tmp_dir),
				n_mers=cfg["n_mers"],
				n_cores=cfg["n_cores_per_gpu"],
				methyl_caller=cfg["methylcaller"],
				output_format=cfg["output_format"],
				output_compression=cfg["output_compression"],
				save_mode=cfg["save_mode"],
				seed=cfg["seed"],
			)

			data_path = _expected_data_path(tmp_dir, cfg["output_format"], cfg["output_compression"])
			if not data_path.exists():
				raise FileNotFoundError(f"Bulk data file not found: {data_path}")

			job_out_dir = Path(tempfile.mkdtemp(prefix=f"{output_stem}_", dir=cfg["output_dir"]))
			args = Namespace(
				input_data=str(data_path),
				model_dir=cfg["model_dir"],
				output_path=str(job_out_dir),
				batch_size=cfg["batch_size"],
				save_logit=cfg["save_logit"],
				adjustment=cfg["adjustment"],
			)
			print(f"[gpu {gpu_id}] deconvolute {bam_path} -> {job_out_dir}", flush=True)
			run_deconvolute(args)

			res_path = job_out_dir / "res.csv.gz"
			if not res_path.exists():
				raise FileNotFoundError(f"Expected res.csv.gz missing: {res_path}")
			res_out = Path(cfg["output_dir"]) / f"{output_stem}_res.csv.gz"
			shutil.copy2(res_path, res_out)

			deconv_path = job_out_dir / "deconvolution.csv"
			if not deconv_path.exists():
				raise FileNotFoundError(f"Expected deconvolution output missing: {deconv_path}")
			deconv_out = Path(cfg["output_dir"]) / f"{output_stem}_deconvolution.csv"
			shutil.copy2(deconv_path, deconv_out)

			fi_path = job_out_dir / "FI.csv"
			if not fi_path.exists():
				raise FileNotFoundError(f"Expected FI.csv missing: {fi_path}")
			fi_out = Path(cfg["output_dir"]) / f"{output_stem}_FI.csv"
			shutil.copy2(fi_path, fi_out)

			if not cfg["keep_all_outputs"]:
				shutil.rmtree(job_out_dir, ignore_errors=True)

			if not cfg["keep_tmp"]:
				shutil.rmtree(tmp_dir, ignore_errors=True)

			result_queue.put((bam_path, True, str(deconv_out)))
		except Exception as exc:
			result_queue.put((bam_path, False, str(exc)))


def main():
	args = _parse_args()
	bams = _collect_bams(args.input_bam_dir, args.bam_file)
	if not bams:
		print("No BAM files provided. Use --bam_file or --input_bam_dir.", file=sys.stderr)
		return 2

	gpu_ids = _parse_gpu_list(args.gpus)
	if not gpu_ids:
		print("No GPUs detected. Set --gpus or CUDA_VISIBLE_DEVICES.", file=sys.stderr)
		return 2

	model_dir = _sanitize_model_dir(args.model_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	name_map = _build_output_name_map(bams)
	cfg = {
		"model_dir": model_dir,
		"f_dmr": args.f_dmr,
		"f_ref": args.f_ref,
		"output_dir": str(output_dir),
		"tmp_root": args.tmp_root,
		"n_mers": args.n_mers,
		"n_cores_per_gpu": args.n_cores_per_gpu,
		"methylcaller": args.methylcaller,
		"output_format": args.output_format,
		"output_compression": args.output_compression,
		"save_mode": args.save_mode,
		"batch_size": args.batch_size,
		"save_logit": args.save_logit,
		"adjustment": args.adjustment,
		"seed": args.seed,
		"keep_tmp": args.keep_tmp,
		"keep_all_outputs": args.keep_all_outputs,
	}

	ctx = get_context("spawn")
	job_queue = ctx.Queue()
	result_queue = ctx.Queue()

	workers = []
	for gpu_id in gpu_ids:
		proc = ctx.Process(target=_worker, args=(gpu_id, job_queue, result_queue, cfg))
		proc.start()
		workers.append(proc)

	for bam in bams:
		job_queue.put((bam, name_map[bam]))
	for _ in workers:
		job_queue.put(None)

	failures = 0
	for _ in bams:
		bam_path, ok, msg = result_queue.get()
		if ok:
			print(f"[done] {bam_path} -> {msg}", flush=True)
		else:
			failures += 1
			print(f"[failed] {bam_path}: {msg}", file=sys.stderr, flush=True)

	for proc in workers:
		proc.join()

	return 1 if failures else 0


if __name__ == "__main__":
	sys.exit(main())
