import visdom
import time
import numpy as np

class Visualizer(object):

    def __init__(self, env="default", **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.env = env

        self.index = {}
        self.log_text = ""

    def reinit(self, env="default", **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.env = env
        return self

    def delete_env(self, env):
        self.vis.delete_env(env=env)

    def plot_doubel_line(self, title, xlabel, ylabel, legend, update_type, x, y, win):
        """
        draw the specific line

        :param title:
        :param xlabel: x axis
        :param ylabel: y axis
        :param legend:
        :param update_type:
        :param x:
        :param y:
        :param win:
        :return:
        """
        self.vis.line(
            X=np.column_stack((np.array([x]), np.array([x]))),
            Y=np.column_stack((np.array([y[0]]), np.array([y[1]]))),
            update=update_type,
            win=win,
            opts=dict(
                title=title,
                legend=[legend[0], legend[1]],
                xlabel=xlabel,
                ylabel=ylabel,
                linecolor=np.array([
                    [0, 191, 255],
                    [0, 255, 0]
                ]),
                dash=np.array(['solid', 'dash'])
            )
        )


    def plot_singel_line(self, title, xlabel, ylabel, legend, update_type, x, y, win):
        self.vis.line(
            X=np.array(np.array([x])),
            Y=np.array(np.array([y])),
            update=update_type,
            win=win,
            opts=dict(
                title=title,
                xlabel=xlabel,
                ylabel=ylabel
            )
        )

    # images function
    def img(self, name, img_, **kwargs):

        self.vis.images(
            img_.cpu().numpy(),
            win=(name),
            opts=dict(title=name),
            **kwargs
        )

    # log function
    def log(self, info, win="log_text"):
        self.log_text += ("[{time}] {info} <br>".format(
            time=time.strftime("%m%d_%H%M%S"), info=info))

        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
