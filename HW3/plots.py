import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

# Plotting taken from augmented-neural-odes paper. Thanks to the authors for a great implementation.


categorical_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

all_categorical_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                          '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                          '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                          '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def vector_field_plt(odefunc, num_points, timesteps, inputs=None, targets=None,
                     model=None, augmented=False, h_min=-2., h_max=2., t_max=1., extra_traj=[],
                     save_fig=''):
    """For a 1 dimensional odefunc, returns the vector field associated with the
    function.

    Parameters
    ----------
    odefunc : ODEfunc instance
        Must be 1 dimensional ODE. I.e. dh/dt = f(h, t) with h being a scalar.

    num_points : int
        Number of points in h at which to evaluate f(h, t).

    timesteps : int
        Number of timesteps at which to evaluate f(h, t).

    inputs : torch.Tensor or None
        Shape (num_points, 1). Input points to ODE.

    targets : torch.Tensor or None
        Shape (num_points, 1). Target points for ODE.

    model : anode.models.ODEBlock instance or None
        If model is passed as argument along with inputs, it will be used to
        compute the trajectory of each point in inputs and will be overlayed on
        the plot.

    h_min : float
        Minimum value of hidden state h.

    h_max : float
        Maximum value of hidden state h.

    t_max : float
        Maximum time to which we solve ODE.

    extra_traj : list of tuples
        Each tuple contains a list of numbers corresponding to the trajectory
        and a string defining the color of the trajectory. These will be dotted.
    """
    if augmented:
        t, hidden, dtdt, dhdt = augmented_ode_grid(odefunc, num_points, timesteps, h_min=h_min, h_max=h_max, t_max=t_max)
    else:
        t, hidden, dtdt, dhdt = ode_grid(odefunc, num_points, timesteps, h_min=h_min, h_max=h_max, t_max=t_max)
    # Create meshgrid and vector field plot
    t_grid, h_grid = np.meshgrid(t, hidden, indexing='ij')
    plt.quiver(t_grid, h_grid, dtdt, dhdt, width=0.004, alpha=0.6)

    # Optionally add input points
    if inputs is not None:
        if targets is not None:
            color = ['red' if targets[i, 0] > 0 else 'blue' for i in range(len(targets))]
        else:
            color = 'red'
        # Input points are defined at t=0, i.e. at x=0 on the plot
        plt.scatter(x=[0] * len(inputs), y=inputs[:, 0].cpu().detach().numpy(), c=color, s=80)

    # Optionally add target points
    if targets is not None:
        color = ['red' if targets[i, 0] > 0 else 'blue' for i in range(len(targets))]
        # Target points are defined at t=1, i.e. at x=1 on the plot
        plt.scatter(x=[t_max] * len(targets), y=targets[:, 0].cpu().detach().numpy(), c=color,
                    s=80)

    if model is not None and inputs is not None:
        color = ['red' if targets[i, 0] > 0 else 'blue' for i in range(len(targets))]
        for i in range(len(inputs)):
            init_point = inputs[i:i+1]
            trajectory = model.trajectory(timesteps, init_point)
            plt.plot(t, trajectory[:, 0, 0].cpu().detach().numpy(), c=color[i],
                     linewidth=2)

    if len(extra_traj):
        for traj, color in extra_traj:
            num_steps = len(traj)
            t_traj = [t_max * float(i) / (num_steps - 1) for i in range(num_steps)]
            plt.plot(t_traj, traj, c=color, linestyle='--', linewidth=2)
            plt.scatter(x=t_traj[1:], y=traj[1:], c=color, s=20)

    plt.xlabel("t")
    plt.ylabel("h(t)")

    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=400, bbox_inches='tight')
        plt.clf()
        plt.close()
        
class Arrow3D(FancyArrowPatch):
    """Class used to draw arrows on 3D plots. Taken from:
    https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def ode_grid(odefunc, num_points, timesteps, h_min=-2., h_max=2., t_max=1.):
    """For a 1 dimensional odefunc, returns the points and derivatives at every
    point on a grid. This is useful for plotting vector fields.

    Parameters
    ----------
    odefunc : anode.models.ODEfunc instance
        Must be 1 dimensional ODE. I.e. dh/dt = f(h, t) with h being a scalar.

    num_points : int
        Number of points in h at which to evaluate f(h, t).

    timesteps : int
        Number of timesteps at which to evaluate f(h, t).

    h_min : float
        Minimum value of hidden state h.

    h_max : float
        Maximum value of hidden state h.

    t_max : float
        Maximum time for ODE solution.
    """
    # Vector field is defined at every point (t[i], hidden[j])
    t = np.linspace(0., t_max, timesteps)
    hidden = np.linspace(h_min, h_max, num_points)
    # Vector at each point in vector field is (dt/dt, dh/dt)
    dtdt = np.ones((timesteps, num_points))  # dt/dt = 1
    # Calculate values of dh/dt using odefunc
    dhdt = np.zeros((timesteps, num_points))
    for i in range(len(t)):
        for j in range(len(hidden)):
            # Ensure h_j has shape (1, 1) as this is expected by odefunc
            h_j = torch.Tensor([hidden[j]]).unsqueeze(0)
            dhdt[i, j] = odefunc(t[i], h_j)
    return t, hidden, dtdt, dhdt

def augmented_ode_grid(odefunc, num_points, timesteps, h_min=-2., h_max=2., t_max=1.):
    """For a 1 dimensional odefunc, returns the points and derivatives at every
    point on a grid. This is useful for plotting vector fields.

    Parameters
    ----------
    odefunc : anode.models.ODEfunc instance
        Must be 1 dimensional ODE. I.e. dh/dt = f(h, t) with h being a scalar.

    num_points : int
        Number of points in h at which to evaluate f(h, t).

    timesteps : int
        Number of timesteps at which to evaluate f(h, t).

    h_min : float
        Minimum value of hidden state h.

    h_max : float
        Maximum value of hidden state h.

    t_max : float
        Maximum time for ODE solution.
    """
    # Vector field is defined at every point (t[i], hidden[j])
    t = np.linspace(0., t_max, timesteps)
    hidden = np.linspace(h_min, h_max, num_points)
    # Vector at each point in vector field is (dt/dt, dh/dt)
    dtdt = np.ones((timesteps, num_points))  # dt/dt = 1
    # Calculate values of dh/dt using odefunc
    dhdt = np.zeros((timesteps, num_points))
    for i in range(len(t)):
        for j in range(len(hidden)):
            # Ensure h_j has shape (1, 1) as this is expected by odefunc
            h_j = torch.Tensor([hidden[j]]).unsqueeze(0)
            aug = torch.zeros_like(h_j)
            aug_inp = torch.cat([h_j, aug], 1)
            dhdt[i, j] = (odefunc(t[i], aug_inp)[:, 0])
    return t, hidden, dtdt, dhdt



def get_feature_history(trainer, dataloader, inputs, targets, num_epochs):
    """Helper function to record feature history while training a model. This is
    useful for visualizing the evolution of features.

    trainer : anode.training.Trainer instance

    dataloader : torch.utils.DataLoader

    inputs : torch.Tensor
        Tensor of shape (num_points, num_dims) containing a batch of data which
        will be used to visualize the evolution of the model.

    targets : torch.Tensor
        Shape (num_points, 1). The targets of the data in inputs.

    num_epochs : int
        Number of epochs to train for.
    """
    feature_history = []
    # Get features at beginning of training
    features, _ = trainer.model(inputs, return_features=True)
    feature_history.append(features.detach())

    for i in range(num_epochs):
        trainer.train(dataloader, 1)
        features, _ = trainer.model(inputs, return_features=True)
        feature_history.append(features.detach())

    return feature_history


def get_square_aspect_ratio(plt_axis):
    return np.diff(plt_axis.get_xlim())[0] / np.diff(plt_axis.get_ylim())[0]
