import mujoco
from dm_control import mjcf

from omegaconf import DictConfig

from stac_mjx import controller as ctrl

"""
Notes: finished modifying create_body_sites. next step would be modifying
part_opt_setup
"""


class STAC:
    def __init__(self, stac_config: DictConfig, model_config: DictConfig) -> None:
        # Gettings paths
        fit_path = stac_config.paths.fit_path
        transform_path = stac_config.paths.transform_path
        mjcf_path = stac_config.paths.xml

        # Set up mjcf
        root = mjcf.from_path(mjcf_path)
        self.physics, self.mj_model, self.site_index_map, self.part_names = (
            ctrl.create_body_sites(root, model_config.SCALE_FACTOR)
        )
        ctrl.part_opt_setup(self.physics)

        self.mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[stac_config.mujoco.solver.lower()]

        self.mj_model.opt.iterations = stac_config.mujoco.iterations
        self.mj_model.opt.ls_iterations = stac_config.mujoco.ls_iterations
        # Runs faster on GPU with this
        self.mj_model.opt.jacobian = 0  # dense
