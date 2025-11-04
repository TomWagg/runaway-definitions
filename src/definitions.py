import numpy as np
import astropy.units as u


def get_theorist_runaways(p, teff_thresh=30000, vel_thresh=30, near_thresh=3 * u.kpc):
    """Identify runaway stars in a population of binaries based on theorist criteria

    For theorists, we consider the system immediately after the first supernova.
    Either of the following two cases need to be met:
        - System is bound after SN (`sep > 0`), companion has `teff > 30kK`, `vsys_1_total > 30 km/s`
        - Or system is disrupted after SN (`sep < 0`), ejected companion has `teff > 30kK`, `vsys_2_total > 30 km/s`

    In both cases the system must be nearby to the sun (within 3 kpc by default).

    Parameters
    ----------
    p : :class:`cogsworth.pop.Population`
        A population of binaries
    teff_thresh : `int`, optional
        Temperature threshold in Kelvin, by default 30000
    vel_thresh : `int`, optional
        Runaway velocity threshold in km/s, by default 30
    near_thresh : `astropy.units.Quantity` [length], optional
        Threshold for whether a system is nearby, by default 3*u.kpc

    Returns
    -------
    theorist_runaway_pop : :class:`cogsworth.pop.Population`
        A population of runaway stars
    theorist_masks : `dict`
        A dictionary of masks used to identify runaway stars
    """
    # get the first kick associated with each binary (ignoring any that didn't have a kick)
    first_kicks = p.kick_info[p.kick_info["star"] != 0].drop_duplicates(subset="bin_num", keep="first")

    # get the bpp rows after the first kick
    first_kick_bpp = p.bpp[p.bpp["evol_type"].isin([15, 16])].drop_duplicates(subset="bin_num", keep="first")
    after_sn_1 = p.bpp.iloc[first_kick_bpp["row_ind"] + 1]

    # track the sun's location
    sun_loc = np.array([8.122, 0, 0]) * u.kpc

    # build some masks for the primary and secondary
    theorist_masks = {
        "co": {i: after_sn_1[f"kstar_{i}"].isin([13, 14]) for i in [1, 2]},             # is a compact object
        "ms": {i: after_sn_1[f"kstar_{i}"] <= 1 for i in [1, 2]},                       # is a MS star
        "hot": {i: after_sn_1[f"teff_{i}"] >= teff_thresh for i in [1, 2]},             # is above T thresh
        "fast": {i: first_kicks[f"vsys_{i}_total"] >= vel_thresh for i in [1, 2]},      # is fast enough
        "bound": after_sn_1["sep"] > 0.0,                                               # remained bound
        "disrupted": after_sn_1["sep"] < 0.0,                                           # got disrupted
        "nearby": {}
    }

    # mask for whether a binary had an SN
    had_sn = np.isin(p.bin_nums, first_kick_bpp["bin_num"])

    # construct a distance mask for every position
    dist_mask = np.linalg.norm(p.final_pos - sun_loc, axis=1) < near_thresh

    primary_nearby = dist_mask[:len(p)]
    secondary_nearby = dist_mask[:len(p)]
    secondary_nearby[p.disrupted] = dist_mask[len(p):]

    theorist_masks["nearby"][1] = primary_nearby[had_sn]
    theorist_masks["nearby"][2] = secondary_nearby[had_sn]

    # several cases could look like runaways
    # a) remained bound, one star is CO, other is MS, hot, fast, and nearby
    bound_runaways = (
        theorist_masks["bound"]
        & theorist_masks["fast"][1]
        & theorist_masks["nearby"][1]
        & (
            (theorist_masks["co"][1] & theorist_masks["ms"][2] & theorist_masks["hot"][2])
            | (theorist_masks["co"][2] & theorist_masks["ms"][1] & theorist_masks["hot"][1])
          )
    )

    # a) or, far more likely, got disrupted, and one star is MS, hot, fast, and nearby
    disrupted_runaways = (
        theorist_masks["disrupted"]
        & ((theorist_masks["fast"][1] & theorist_masks["ms"][1]
            & theorist_masks["hot"][1] & theorist_masks["nearby"][1])
           | (theorist_masks["fast"][2] & theorist_masks["ms"][2]
              & theorist_masks["hot"][2] & theorist_masks["nearby"][2]))
    )

    print(f"Found {np.sum(bound_runaways)} bound runaway candidates and {np.sum(disrupted_runaways)} disrupted runaway candidates")

    theorist_runaway_pop = p[first_kicks["bin_num"].values[bound_runaways | disrupted_runaways].astype(int)]

    return theorist_runaway_pop, theorist_masks


def get_final_vel_cylindrical(pop):
    x, y = pop.final_pos[:, 0], pop.final_pos[:, 1]
    vx, vy, vz = pop.final_vel[:, 0], pop.final_vel[:, 1], pop.final_vel[:, 2]

    # compute cylindrical radius and angle
    R = np.sqrt(x*x + y*y)

    # radial and tangential components in the disk plane
    v_R = (x*vx + y*vy) / R
    v_T = (-y*vx + x*vy) / R

    final_vel_cylindrical = np.vstack([v_R, v_T, vz]).T
    return final_vel_cylindrical


def get_observer_population(p, teff_thresh=30000, vel_thresh=30, near_thresh=3 * u.kpc, v_circ=None):
    """Identify runaway stars in a population of binaries based on observer criteria

    Parameters
    ----------
    p : :class:`cogsworth.pop.Population`
        A population of binaries
    teff_thresh : `int`, optional
        Temperature threshold in Kelvin, by default 30000
    vel_thresh : `int`, optional
        Runaway velocity threshold in km/s, by default 30
    near_thresh : `astropy.units.Quantity` [length], optional
        Threshold for whether a system is nearby, by default 3*u.kpc
    v_circ : :class:`astropy.units.Quantity` [velocity], optional
        Circular velocity at the location of each system at present day, by default None

    Returns
    -------
    observer_runaway_pop : :class:`cogsworth.pop.Population`
        A population of runaway stars
    observer_masks : dict
        A dictionary of masks used to identify runaway stars
    """
    # compute the circular velocity at each star's position if not provided
    if v_circ is None:
        v_circ = p.galactic_potential.circular_velocity(p.final_pos.T)

    # construct a distance mask for every position
    sun_loc = np.array([8.122, 0, 0]) * u.kpc
    dist_mask = np.linalg.norm(p.final_pos - sun_loc, axis=1) < near_thresh
    primary_nearby = dist_mask[:len(p)]
    secondary_nearby = dist_mask[:len(p)]
    secondary_nearby[p.disrupted] = dist_mask[len(p):]

    # things that are moving fast
    final_vel_cyl = get_final_vel_cylindrical(p)
    rel_vel = np.linalg.norm([final_vel_cyl[:, 0], final_vel_cyl[:, 1] - v_circ, final_vel_cyl[:, 2]], axis=0)
    fast = rel_vel > vel_thresh
    primary_fast = fast[:len(p)]
    secondary_fast = fast[:len(p)].copy()
    secondary_fast[p.disrupted] = fast[len(p):]

    # build some masks for the primary and secondary
    observer_masks = {
        "co": {i: p.final_bpp[f"kstar_{i}"].isin([13, 14]) for i in [1, 2]},        # is a compact object
        "ms": {i: p.final_bpp[f"kstar_{i}"] <= 1 for i in [1, 2]},                  # is a MS star
        "hot": {i: p.final_bpp[f"teff_{i}"] >= teff_thresh for i in [1, 2]},        # is above T thresh
        "nearby": {1: primary_nearby, 2: secondary_nearby},
        "fast": {1: primary_fast, 2: secondary_fast}
    }

    # define O star masks and bound systems
    observer_masks["o_star"] = {i: observer_masks["ms"][i] & observer_masks["hot"][i] for i in [1, 2]}

    # for observers, we just need to find a star that's an O star, nearby and moving fast
    # potential runaways could come in a few different categories

    disrupted_runaways = (
        (p.final_bpp["sep"] < 0.0)
        & ((observer_masks["o_star"][1] & observer_masks["nearby"][1] & observer_masks["fast"][1])
           | (observer_masks["o_star"][2] & observer_masks["nearby"][2] & observer_masks["fast"][2]))
    )

    bound_runaways = (
        (p.final_bpp["sep"] > 0.0)
        & observer_masks["fast"][1]
        & observer_masks["nearby"][1]
        & ((observer_masks["co"][1] & observer_masks["o_star"][2])
           | (observer_masks["co"][2] & observer_masks["o_star"][1]))
    )

    merger_runaways = (
        (p.final_bpp["sep"] == 0.0)
        & (observer_masks["o_star"][1] | observer_masks["o_star"][2])
        & observer_masks["fast"][1]
        & observer_masks["nearby"][1]
    )

    print(f"Found {np.sum(disrupted_runaways)} disrupted runaway candidates, "
          f"{np.sum(bound_runaways)} bound runaway candidates, and "
          f"{np.sum(merger_runaways)} merger runaway candidates.")

    observer_runaway_nums = p.bin_nums[disrupted_runaways | bound_runaways | merger_runaways]
    observer_runaway_nums = np.unique(observer_runaway_nums)

    return p[observer_runaway_nums], observer_masks
