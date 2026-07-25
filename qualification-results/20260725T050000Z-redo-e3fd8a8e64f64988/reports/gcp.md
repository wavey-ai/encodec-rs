# GCP

Status: blocked.

The project is `steadfast-slate-498623-r2`. A temporary `n2-standard-4` worker was created in `us-west1-a` and deleted after the package transfer failed. No GCP CPU row ran.

GCP GPU provisioning was checked in `us-central1-a` and `us-west1-a`. The project-wide `GPUS_ALL_REGIONS` quota is zero, so no CUDA worker could start.
