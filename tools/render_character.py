"""One-click character render.  Drop this file into an empty folder and run it:

    python render_character.py              headshot + full body: front, 3/4 L, 3/4 R, side, back
    python render_character.py --headonly   headshot only (5 views)
    python render_character.py --presets human_f,human_m   (body turnaround for several bases)
    python render_character.py --no-split   mesh the body in one process (if the parallel mesher misbehaves)
    python render_character.py --preview    fast brow/lash close-up (needs one full run first, for the cache)
    python render_character.py --samples 512

What it does, all inside this folder:
  1. makes sure a Python 3.13 is used (the Blender 5.2 'bpy' module is built for 3.13)
  2. creates a private virtual environment (.venv) and installs numpy, pillow, bpy
  3. clones / updates the project from GitHub (branch claude/pensive-volta-a6fl2i)
  4. headshot renders on the GPU WHILE the body meshes on every CPU core, then the body renders on the GPU
     (reuses output/hs_cache_*.npz from an earlier run in this same folder)
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
    work = os.path.join(REPO, "tools", "modelgen")
    if "--preview" in sys.argv:
        out_png = os.path.join(OUT, "lashes_closeup.png")
        run([py, "browlash_preview.py", out_png, "human_f", "64", "--eye"], cwd=work, env=env)
    else:
        cache = os.path.join(OUT, "hs_cache_human_f.npz")
        hs = [py, "headshot.py", os.path.join(OUT, "headshot.png"), "human_f", samples, "--turnaround"]
        if not os.path.exists(cache):
            hs.append("--rebuild")             # first run: sculpt mesh + grow the hair (cached afterwards)
        procs = [subprocess.Popen(hs, cwd=work, env=env)]
        if "--headonly" not in sys.argv:
            ta = [py, "turnaround.py", "--samples", samples]
            for flag in ("--presets", "--workers", "--h", "--res"):
                if flag in sys.argv:
                    ta += [flag, sys.argv[sys.argv.index(flag) + 1]]
            if "--no-split" in sys.argv:
                ta.append("--no-split")
            procs.append(subprocess.Popen(ta, cwd=work, env=env))
        codes = [p.wait() for p in procs]
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
