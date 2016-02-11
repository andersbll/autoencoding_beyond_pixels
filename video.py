import os
import subprocess
import numpy as np
import scipy.misc

try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


class Video(object):
    def __init__(self, filename, fps=15, bitrate=2000, dump_imgs=True):
        self.filename = filename
        self.fps = fps
        self.bitrate = bitrate
        self.dump_imgs = dump_imgs
        if which('ffmpeg') is None and which('avconv') is None:
            self.disable_video = True
        else:
            self.disable_video = False
        dirpath = os.path.split(filename)[0]
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)
        if dump_imgs:
            self.imgs_dir = os.path.splitext(filename)[0]
            if not os.path.exists(self.imgs_dir):
                os.mkdir(self.imgs_dir)
        self.proc = None
        self.img_shape = None
        self.n_frames = 0

    def _start_proc(self):
        if len(self.img_shape) == 3 and self.img_shape[2] == 3:
            pix_fmt = 'rgb24'
        else:
            pix_fmt = 'gray'
        if which('ffmpeg') is None:
            encoder = 'avconv'
        else:
            encoder = 'ffmpeg'
        cmd = [
            encoder,
            '-y',
            '-loglevel', 'error',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '%dx%d' % (self.img_shape[1], self.img_shape[0]),
            '-pix_fmt', pix_fmt,
            '-r', '%.02f' % self.fps,
            '-i', '-', '-an',
            self.filename,
        ]
        popen_params = {"stdout": DEVNULL,
                        "stderr": subprocess.PIPE,
                        "stdin": subprocess.PIPE}
        self.proc = subprocess.Popen(cmd, **popen_params)

    def append(self, frame):
        if frame.dtype.kind == 'f':
            frame -= np.min(frame)
            frame /= np.max(frame)/255.
            frame = frame.astype(np.uint8)
        if frame.shape[0] % 2 != 0:
            frame = np.insert(frame, frame.shape[0], 0, axis=0)
        if frame.shape[1] % 2 != 0:
            frame = np.insert(frame, frame.shape[1], 0, axis=1)
        if self.dump_imgs:
            img_path = os.path.join(self.imgs_dir, '%.5d.png' % self.n_frames)
            scipy.misc.imsave(img_path, frame)
        if self.img_shape is None:
            self.img_shape = frame.shape
            if frame.ndim == 3 and frame.shape[2] not in [1, 3]:
                raise ValueError('invalid # of channels: %i' % frame.shape[2])
        elif self.img_shape != frame.shape:
            raise ValueError('frame shape mismatch')
        self.n_frames += 1

        if self.disable_video:
            return

        if self.proc is None:
            self._start_proc()
        try:
            self.proc.stdin.write(frame.tostring())
            self.proc.stdin.flush()
        except IOError:
            raise RuntimeError('Failed writing frame to video:\n %s '
                               % self.proc.stderr.read())

    def __del__(self):
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait()
            del self.proc
