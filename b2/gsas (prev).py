import os
import sys
import numpy as np
from typing import List


with open(os.devnull, "w") as f:
    sys.stdout = f

    gsas_bin = os.path.join(sys.base_prefix, "GSASII/bindist")
    gsas_path = os.path.join(sys.base_prefix, "GSASII")

    for p in (gsas_path, gsas_bin):
        if p not in sys.path:
            sys.path.insert(0, p)

    import GSASIIstrIO as G2stIO
    import GSASIIIO as G2IO
    import GSASIIscriptable as G2sc
    import GSASIIstrMath
    import GSASIImapvars as G2mv

    # note: build fortran components
    # cd ${GSAS_DIR}/fsource && make
    import pypowder as pyd
    import pydiffax as pyx

    sys.stdout = sys.__stdout__


class GSASModel:
    def __init__(self, project_file="TaTisim.gpx"):
        """Load a GSAS structural model from a project file.

        ```python
        m = gsas.GSASModel("data/TaTisim.gpx")
        y = m.forward()
        ```
        """

        controls = G2stIO.GetControls(project_file)
        histograms, phases = G2stIO.GetUsedHistogramsAndPhases(project_file)

        phase_data = G2stIO.GetPhaseData(phases, Print=False)
        (
            Natoms,
            atomIndx,
            phase_variables,
            phases_dict,
            pawley_lookup,
            FFtables,
            BLtables,
            *rest,
        ) = phase_data

        controls["atomIndx"] = atomIndx
        controls["Natoms"] = Natoms
        controls["FFtables"] = FFtables
        controls["BLtables"] = BLtables

        # rigid body variables: nothing i.e. {}
        rigidbodies = G2stIO.GetRigidBodies(project_file)
        rb_ids = rigidbodies.get("RBIds", {"Vector": [], "Residue": []})
        rb_variables, rb_data = G2stIO.GetRigidBodyModels(rigidbodies, Print=False)

        hap_variables, hap_data, control_data = G2stIO.GetHistogramPhaseData(
            phases, histograms, Print=False
        )
        controls.update(control_data)

        hist_variables, hist_data, control_data = G2stIO.GetHistogramData(
            histograms, Print=False
        )
        controls.update(control_data)

        self.controls = controls

        # collect all model variables in one place
        self.variables = rb_variables + phase_variables + hap_variables + hist_variables

        # construct a dictionary of all model parameters
        params = {}
        params.update(rb_data)
        params.update(phases_dict)
        params.update(hap_data)
        params.update(hist_data)
        self.params = params

        G2stIO.GetFprime(controls, histograms)

        # convenient references to model components
        self.phases = phases
        self.hist = list(histograms.values())[0]
        self.histlist = histograms
        self.tth = self.hist["Data"][0][0:-1]
        self.pawley_lookup = pawley_lookup

    def update(
        self, site_occupancies=[0.5, 0.5, 0.5, 0.5], grainsize=None, mustrain=None
    ):
        """Update parameters consistently.

        grainsize in microns

        1. mutate values in self.hist and self.phases
        2. G2stIO.GetHistogramPhaseData, G2stIO.GetHistogramData, G2stIO.GetPhaseData
        3. update self.params

        size: '0:0:Size;i'
        """

        # set cubic lattice parameter...
        # mutate internal data in place...
        # cell = [False, 3.38, 3.38, 3.38, 90.0, 90.0, 90.0, 38.61447199999999]
        phase_data = self.phases[list(self.phases.keys())[0]]

        # update histogram variables...
        # grab the first histogram
        _, _hist = next(iter(phase_data["Histograms"].items()))

        if grainsize is not None:
            # update isotropic size
            _hist["Size"][1][0] = grainsize

        if mustrain is not None:
            # update isotropic mustrain
            _hist["Mustrain"][1][0] = mustrain

        hap_variables, hap_data, control_data = G2stIO.GetHistogramPhaseData(
            self.phases, self.histlist, Print=False
        )
        hist_variables, hist_data, control_data = G2stIO.GetHistogramData(
            self.histlist, Print=False
        )

        phase_data = G2stIO.GetPhaseData(self.phases, Print=False)
        (
            Natoms,
            atomIndx,
            phase_variables,
            phases_dict,
            pawley_lookup,
            FFtables,
            BLtables,
            *rest,
        ) = phase_data

        # compute site occupancy values from order parameters
        # RA = 0.5 * S + 0.5
        # phase_data[3]["0::Afrac:0"] = 1 - RA
        # phase_data[3]["0::Afrac:1"] = 1 - RA
        # phase_data[3]["0::Afrac:2"] = RA
        # phase_data[3]["0::Afrac:3"] = RA
        for site_idx, occu in enumerate(site_occupancies):
            phase_data[3][f"0::Afrac:{site_idx}"] = occu

        phases_dict = phase_data[3]

        self.params.update(hap_data)
        self.params.update(hist_data)
        self.params.update(phases_dict)

    def forward(
        self, site_occupancies=[0.5, 0.5, 0.5, 0.5], grainsize=None, mustrain=None
    ):
        """Compute model powder profile with current phase and histogram settings."""

        self.update(
            site_occupancies=site_occupancies, grainsize=grainsize, mustrain=mustrain
        )

        profile, background = GSASIIstrMath.getPowderProfile(
            self.params,
            self.tth,
            self.variables,
            self.hist,
            self.phases,
            self.controls,
            self.pawley_lookup,
        )

        return profile.data + 600 * background.data

    def derivative(self, variables: List[str]):
        """Compute powder profile derivatives wrt `variables`.

        ex. variables = ["0::A0"]

        note: call `forward` first to make sure all internals are updated
        """

        args = [
            self.params,
            self.tth,
            self.variables,
            self.hist,
            self.phases,
            {},
            self.controls,
            self.pawley_lookup,
            variables,
        ]

        d = GSASIIstrMath.getPowderProfileDerv(args)
        return d
