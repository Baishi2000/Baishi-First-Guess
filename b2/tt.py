import theano
import numpy as np
import pandas as pd

import theano.tensor as tt
from theano.graph.op import Op

from b2.gsas import GSASModel


class GSASWrapper(Op):
    """Wrap GSAS forward model in theano op.

    ```python
    model = GSASModel("project.gpx")
    wrapper = GSASWrapper(model)

    with pm.Model() as order_param_model:
         S = pm.Uniform("S", 0.0, 0.5, testval=0.25)
         RA = 0.5 * S + 0.5

         raw_sz = pm.Lognormal("raw_sz", mu=0.0, sigma=1.0, testval=1.0)
         sz = pm.Deterministic("sz", 5e-3 * raw_sz)
         raw_mustrain = pm.Lognormal("raw_mustrain", mu=0.0, sigma=1.0, testval=1.0)
         mustrain = pm.Deterministic("mustrain", 5e4 * raw_mustrain)

         # pack inputs into theano tensor...
         x = tt.as_tensor_variable([1-RA, 1-RA, RA, RA, sz, mustrain])
         ycalc = pm.Deterministic("ycalc", gsas_eval(x))
         yobs = pm.Normal("yobs", mu=ycalc, sd=tt.sqrt(ycalc), shape=(1400,), observed=yy)
         trace = pm.sample(10, tune=10, return_inferencedata=True)
    ```
    """

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    # just site occupancy values...
    inputs = [f"0::Afrac:{site_idx}" for site_idx in range(4)] + [
        "0:0:Size;i",
        "0:0:Mustrain;i",
    ]

    def __init__(self, model):
        self.model = model

    def perform(self, node, inputs, outputs):
        (x,) = inputs

        # hacky -- there are four sites!
        S = x[:4]
        sz, mustrain = x[4], x[5]

        y = self.model.forward(site_occupancies=S, grainsize=sz, mustrain=mustrain)
        outputs[0][0] = y

    def grad(self, inputs, grads):
        """Collect gradients with GSAS backend call."""
        grad_op = GSASGrad(self)
        return [grad_op(inputs, grads[0])]


class GSASGrad(Op):
    def __init__(self, model):
        self.model = model

    def make_node(self, x, g):
        x = tt.as_tensor_variable(x)
        g = tt.as_tensor_variable(g)
        return tt.Apply(self, [x, g], [g.type()])

    def perform(self, node, inputs, outputs):

        x = inputs[0]

        g = inputs[1]

        self.model.model.forward(site_occupancies=x[0])
        _, grad_data = self.model.model.derivative(self.model.inputs)

        # # Note: GSAS optimizer uses metric tensor components
        # # instead of direct lattice parameters (e.g. A0 instead of a)
        # # if we want to model direct lattice parameters, need to apply the chain rule
        # # to get the derivative of the pattern w.r.t. the lattice parameters...
        # dfdA = grad_data["0::A0"]
        # dfda = dfdA * -2 * (a ** -3)
        dfdx = pd.DataFrame(grad_data).values

        outputs[0][0] = dfdx.T.dot(g)
