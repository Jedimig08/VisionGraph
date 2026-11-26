# VisionGraph on Termux

This branch contains the Termux-specific files. The PC client (`cv_stream.py`) and the
shared computer-vision modules remain unchanged.

## 1. Set up sshd on the phone

Open Termux and run:

```sh
pkg update
pkg install -y openssh
passwd
sshd
whoami
ifconfig | grep inet
```

`sshd` listens on port **8022** by default. `whoami` prints your username (for example
`u0_a351`) and `ifconfig` prints the phone's IP (for example `192.168.1.67`).

From the PC:

```sh
ssh -p 8022 <username>@<phone-ip>
```

If password login is refused, edit `$PREFIX/etc/ssh/sshd_config` so it contains
`PasswordAuthentication yes`, then run `sshd` again.

To start sshd automatically after Termux is killed, either install the Termux:Boot app
and add a boot script that runs `sshd`, or add this to `~/.bashrc`:

```sh
if ! pgrep -x sshd >/dev/null 2>&1; then sshd; fi
```

## 2. Install the code and dependencies

Copy this branch to the phone (for example with `scp -P 8022`), then run:

```sh
cd ~/VisionGraph/termux
chmod +x install.sh
./install.sh
```

The installer installs only what is missing, then copies the Python files into
`$HOME/VisionGraph` (override with `VISIONGRAPH_DIR`) and verifies that `numpy`, `cv2`,
`cv2.aruco`, and `flask` import successfully.

OpenCV on Termux is provided by `opencv-python` from `x11-repo`. Adding `x11-repo` only
enables that package source — it does not install or start an X server, and the Termux
client never calls `cv2.imshow`. The installer adds the repo only when `cv2` is missing.
There is no headless OpenCV package in Termux yet (a request is still open upstream), so
using the packaged `opencv-python` without a display is still the lightest working option.

`zeroconf` is **not** installed or used on the phone. CommandHub is reached directly at
`http://127.0.0.1:8080`; its camera endpoints are `/api/cameras` and
`/stream/{id}?res=WxH&fps=N`.

If the verification step reports that `cv2.aruco` is missing, the installed OpenCV build
does not include the contrib modules. Install a build that provides `cv2.aruco` before
running the vision pipeline; the installer exits with an error in that case.

## 3. CommandHub launcher

The launcher should point at the local processing server (for example
`http://127.0.0.1:5000/`). Clicking **Launch Processing UI** from a laptop saves the URL
and opens `http://<phone-ip>:8080/processing-ui/`, where CommandHub proxies the page
running in Termux. CommandHub's access mode must be `ONLINE` for laptop access.
