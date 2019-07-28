import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab


class Evaluation:
    def __init__(self):
        pass

    @staticmethod
    def _coordinate(array):
        tmp = np.clip(array * 127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return tmp

    def __call__(self, y, c, s, outdir, epoch, testsize):
        pylab.rcParams['figure.figsize'] = (16.0,16.0)
        pylab.clf()

        for i in range(testsize):
            tmp = self._coordinate(c[i])
            pylab.subplot(testsize, testsize, 3 * i + 1)
            pylab.imshow(tmp)
            pylab.axis('off')
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")
            tmp = self._coordinate(s[i])
            pylab.subplot(testsize, testsize, 3 * i + 2)
            pylab.imshow(tmp)
            pylab.axis('off')
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")
            tmp = self._coordinate(y[i])
            pylab.subplot(testsize, testsize, 3 * i + 3)
            pylab.imshow(tmp)
            pylab.axis('off')
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")