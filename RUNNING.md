# Running on Kaggle

GPU training runs on Kaggle's free GPUs. You set up one notebook once. After
that, every run is one click: **Save Version → Save & Run All (Commit)**. The
notebook reads `runs/queue.yaml` from the `main` branch and works through the
jobs in it.

Everything here is free: Kaggle (about 30 GPU hours a week), Hugging Face Hub
(free public storage), and GitHub.

## One-time setup

### 1. Hugging Face token

1. Make a free account at huggingface.co.
2. Go to **Settings → Access Tokens → Create new token**.
3. Pick token type **Write**, give it a name (for example `kaggle-gptlab`), and create it.
4. Copy the token (it starts with `hf_`).

The runner creates two public repos under your account the first time it runs:

- `<your-hf-name>/gptlab-data`: a dataset repo with the token shards (about 5 GB)
- `<your-hf-name>/gptlab-checkpoints`: a model repo with checkpoints

To make them private, set `hf_private: true` in `runs/queue.yaml`. Free
accounts get 100 GB of private storage, which is enough for this project.

### 2. GitHub token

The runner pushes logs to the `results` branch of this repo. It needs a token
that can write to this one repo and nothing else.

1. On GitHub: **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
2. **Repository access**: *Only select repositories* → `MelvTheGoat/LLM`.
3. **Permissions → Repository permissions → Contents**: *Read and write*. Leave everything else as is.
4. Pick an expiry (90 days is fine), generate, and copy the token (it starts with `github_pat_`).

### 3. Kaggle notebook

1. Verify your phone number in your Kaggle account settings. Kaggle needs this before it allows GPUs and internet access.
2. **Create → New Notebook**.
3. **File → Import Notebook**, then either paste this link:
   `https://raw.githubusercontent.com/MelvTheGoat/LLM/main/kaggle/runner.ipynb`
   or upload `kaggle/runner.ipynb` from the repo. (You can also copy the code
   cell from that file into an empty notebook.)
4. In the right-hand panel, under **Session options**:
   - **Accelerator**: `GPU T4 x2`. Do not pick P100: it has no fp16 tensor cores, and recent PyTorch builds may not support it at all.
   - **Internet**: on.
   - **Environment**: *Pin to original environment*. Then every run uses the same PyTorch version, so results stay comparable.
5. **Add-ons → Secrets**:
   - Add a secret named `GH_TOKEN` with the GitHub token.
   - Add a secret named `HF_TOKEN` with the Hugging Face token.
   - Make sure both are ticked (attached) for this notebook.
6. Give the notebook a name (for example `gptlab-runner`) and save it.

## Starting a run

1. Check the accelerator: **GPU T4 x2** for GPU jobs, **None** for CPU jobs (see below).
2. Click **Save Version** (top right), choose **Save & Run All (Commit)**, and click **Save**.
3. That's it. The run happens in the background, so you can close the tab. To
   watch it, open the notebook's **Version history** (or **View Active Events**)
   and open the running version's log.

Each run:

1. clones the `main` branch and installs the requirements;
2. reads `runs/queue.yaml` and picks the first job that can run now;
3. downloads the token data and the latest checkpoint (if any) from Hugging Face;
4. runs the job, resuming from the checkpoint if there is one;
5. while training, uploads a checkpoint every 30 minutes and pushes logs every 10 minutes;
6. stops training 10 hours 40 minutes in, saves, uploads, and pushes the logs. It
   is done well before Kaggle's 12-hour limit. The job is marked `paused` and
   the next run resumes it;
7. starts the next job if there is enough time left.

If a job crashes, the logs and status are still pushed. A crashed job is tried
once more on the next run. After that it waits for a code fix.

### GPU jobs and CPU jobs

Each job in the queue says `hardware: gpu` or `hardware: cpu`. A GPU session
only runs GPU jobs, and a CPU session only runs CPU jobs. CPU sessions
(Accelerator: **None**) are free and don't use your 30 GPU hours, so the
data-building job runs there.

You can run a CPU session and a GPU session at the same time (save one version
with GPU, switch the accelerator, save another). The runner makes sure two
sessions never pick the same job.

## Order of runs

1. **`smoke-2`** (GPU T4 x2, done): ran the whole pipeline on a tiny slice of
   data and measured the speed of six model sizes. Those speeds set the compute
   budget in [EXPERIMENTS.md](EXPERIMENTS.md).
2. **`data-fineweb-edu-16k`** (Accelerator **None**, about 1 to 3 hours): builds
   the real dataset (about 2.6B training tokens) and uploads it to Hugging Face.
   Every training run waits for it, so run it first.
3. **Experiments** (GPU T4 x2), in stages. The queue only ever holds the current
   stage. After each stage, say "check results": the next stage is chosen from
   what the last one showed (for example, the learning rates).

If you start a GPU session while the dataset is still being built, it runs the
jobs that don't need data (the speed benchmark) and then stops.

## Checking results

Logs land on the `results` branch, one folder per job:
`https://github.com/MelvTheGoat/LLM/tree/results/results`

- `status.json`: the job's state (`running`, `paused`, `done`, `diverged` or `failed`), progress, and a message
- `metrics.jsonl`: one line per training step (loss, learning rate, gradient norm, tokens/s, MFU, memory)
- `eval.jsonl`, `debug.jsonl`, `final_eval.json`, `samples.txt`, `train.log`: evaluation, per-layer stats, final numbers, sample text, console output

## How code gets to Kaggle

Kaggle always runs the `main` branch. New code is pushed to the `dev` branch
and reaches `main` when you merge a pull request. Before a run that needs new
code, you will be told to merge first.

## Troubleshooting

| What you see | What it means / what to do |
| --- | --- |
| `Secret GH_TOKEN is missing` | Add the secret under Add-ons → Secrets and tick it for this notebook. |
| `GH_TOKEN is not set` or `git push` fails with 403 | The GitHub token expired or lacks *Contents: Read and write* on this repo. Make a new one. |
| A job shows `running` but no session is running | That session was killed (for example stopped by hand or out of quota). After 30 minutes without a heartbeat, the next run picks the job up again and resumes from the last checkpoint. |
| A job shows `failed` twice | It needs a code fix. Say "check results". |
| `no output for 30 minutes: the command looks stuck` or `time limit reached` | The runner stopped a job that hung or ran past its `max_hours` limit, so it could not eat your GPU hours. Say "check results". |
| `no job can run in this session` | Every job is done, running elsewhere, waiting for another job, or needs the other session type (GPU vs CPU). The log lists the reason for each job. |
| The weekly GPU quota ran out mid-run | Nothing is lost except at most 30 minutes of training. The job resumes next week, or as soon as there is quota. |
| You want to stop a run early | Stop the session in Kaggle. The job resumes next time from the last checkpoint (at most 30 minutes old). |
