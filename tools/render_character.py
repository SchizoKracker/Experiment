"""One-click character render.  Drop this file into an empty folder and run it:

    python render_character.py              headshot + full body: front, 3/4 L, 3/4 R, side, back
    python render_character.py --headonly   headshot only (5 views)
    python render_character.py --presets human_f,human_m   (body turnaround for several bases)
    python render_character.py --no-gpu     mesh on every CPU core instead of the GPU
    python render_character.py --workers 8  cap the CPU worker processes (default: all cores, RAM permitting)
    python render_character.py --preview    fast brow/lash close-up (needs one full run first, for the cache)
    python render_character.py --samples 512

What it does, all inside this folder:
  1. makes sure a Python 3.13 is used (the Blender 5.2 'bpy' module is built for 3.13)
  2. creates a private virtual environment (.venv) and installs numpy, pillow, bpy, cupy (NVIDIA GPU)
  3. clones / updates the project from GitHub (branch claude/pensive-volta-a6fl2i)
  4. builds the models with the shape math on the GPU (CuPy; falls back to every CPU core), then renders
     everything in Cycles on the GPU (OptiX).  Builds are cached in output/ and reused until the model code changes.
  5. opens the output folder
Needs: Python 3.13 and git on PATH (Windows: winget install Python.Python.3.13 Git.Git).
"""
import os
import shutil
import subprocess
import sys
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
REPO_URL = "https://github.com/SchizoKracker/Experiment.git"
BRANCH = "claude/pensive-volta-a6fl2i"
REPO = os.path.join(HERE, "Experiment")
OUT = os.path.join(HERE, "output")
PKGS = ["numpy", "pillow", "bpy==5.2.3"]
GPU_PKGS = ["cupy-cuda12x[ctk]==14.2.0"]      # NVIDIA GPU field evaluation (CUDA runtime comes as pip wheels)


def run(cmd, **kw):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, **kw)


def venv_python():
    return os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV, "bin", "python")


def ensure_python():
    if sys.version_info[:2] == (3, 13):
        return
    # try to relaunch with a 3.13 interpreter
    cands = [["py", "-3.13"]] if os.name == "nt" else [["python3.13"]]
    for c in cands:
        if shutil.which(c[0]):
            try:
                subprocess.check_call(c + ["-c", "import sys; assert sys.version_info[:2] == (3, 13)"])
                sys.exit(subprocess.call(c + [os.path.abspath(__file__)] + sys.argv[1:]))
            except subprocess.CalledProcessError:
                pass
    sys.exit("Python 3.13 is required (bpy 5.2 is built for it).  Windows: winget install Python.Python.3.13")


def ensure_venv():
    py = venv_python()
    if not os.path.exists(py):
        print("creating virtual environment ...", flush=True)
        venv.create(VENV, with_pip=True)
    stamp = os.path.join(VENV, ".deps_ok")
    if not os.path.exists(stamp):
        run([py, "-m", "pip", "install", "--upgrade", "pip"])
        run([py, "-m", "pip", "install"] + PKGS)
        open(stamp, "w").write("ok")
    gstamp = os.path.join(VENV, ".gpu_ok")
    if not os.path.exists(gstamp) and "--no-gpu" not in sys.argv:
        try:
            run([py, "-m", "pip", "install"] + GPU_PKGS)
            open(gstamp, "w").write("ok")
        except subprocess.CalledProcessError:
            print("CuPy install failed: meshing will use every CPU core instead of the GPU", flush=True)
    return py


def ensure_repo():
    if not shutil.which("git"):
        sys.exit("git is required.  Windows: winget install Git.Git")
    if not os.path.isdir(os.path.join(REPO, ".git")):
        run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, REPO])
    else:
        run(["git", "-C", REPO, "fetch", "origin", BRANCH])
        run(["git", "-C", REPO, "checkout", BRANCH])
        run(["git", "-C", REPO, "reset", "--hard", "origin/" + BRANCH])


def main():
    ensure_python()
    py = ensure_venv()
    ensure_repo()
    os.makedirs(OUT, exist_ok=True)
    samples = "256"
    if "--samples" in sys.argv:
        samples = sys.argv[sys.argv.index("--samples") + 1]
    env = dict(os.environ, MODELGEN_OUT=OUT, PYTHONUNBUFFERED="1")
    if "--no-gpu" in sys.argv:
        env["MODELGEN_GPU"] = "0"
    if "--workers" in sys.argv:
        env["MODELGEN_WORKERS"] = sys.argv[sys.argv.index("--workers") + 1]
    work = os.path.join(REPO, "tools", "modelgen")
    if "--preview" in sys.argv:
        out_png = os.path.join(OUT, "lashes_closeup.png")
        run([py, "browlash_preview.py", out_png, "human_f", "64", "--eye"], cwd=work, env=env)
    else:
        cache = os.path.join(OUT, "hs_cache_human_f.npz")
        hs = [py, "headshot.py", os.path.join(OUT, "headshot.png"), "human_f", samples, "--turnaround"]
        if not os.path.exists(cache):
            hs.append("--rebuild")             # first run: sculpt mesh + grow the hair (cached afterwards)
        procs = [hs]
        if "--headonly" not in sys.argv:
            ta = [py, "turnaround.py", "--samples", samples]
            for flag in ("--presets", "--h", "--res"):
                if flag in sys.argv:
                    ta += [flag, sys.argv[sys.argv.index(flag) + 1]]
            if "--no-gpu" in sys.argv:
                ta.append("--no-gpu")
            procs.append(ta)
        # one after the other: each step already saturates the GPU (or every CPU core) on its own
        codes = [subprocess.call(c, cwd=work, env=env) for c in procs]
        if any(codes):
            sys.exit(f"a render step failed (exit codes {codes})")
    print("\ndone ->", OUT)
    try:
        if os.name == "nt":
            os.startfile(OUT)
        elif sys.platform == "darwin":
            subprocess.call(["open", OUT])
    except Exception:
        pass


if __name__ == "__main__":
    main()
