import cogsworth
import astropy.units as u
import numpy as np
import argparse
import gala.potential as gp

import sys
sys.path.append("/mnt/home/twagg/projects/runaway-definitions/src")
import sfh_model


def main():
    parser = argparse.ArgumentParser(description="Generate a population of runaway stars")
    parser.add_argument("-o", "--output-file", type=str, help="Output file to save the population to")
    parser.add_argument('-d', '--dispersion', type=float, help="Velocity dispersion of the runaway stars")
    parser.add_argument('-r', '--radius', type=float, help="Cluster radius")
    parser.add_argument('-n', '--n-per-cluster', type=int, default=100)
    parser.add_argument('-k', '--kick-flag', type=int, default=5)
    parser.add_argument('-c', '--cores', type=int, default=32)

    args = parser.parse_args()

    potential = gp.MilkyWayPotential2022()
    p = cogsworth.pop.Population(10_000_000, final_kstar1=[13, 14],
                                 m1_cutoff=7, processes=args.cores,
                                 galactic_potential=potential,
                                 sfh_model=sfh_model.ClusteredNearSun,
                                 sfh_params={"sfh_model": sfh_model.RecentSB15Annulus,
                                             "sfh_params": {"age_cutoff": 200 * u.Myr,
                                                            "verbose": True,
                                                            "potential": potential,
                                                            "immediately_sample": True},
                                             "near_thresh": 3 * u.kpc,
                                             "cluster_radius": args.radius * u.pc,
                                             "n_per_cluster": args.n_per_cluster,
                                             "velocity_dispersion": args.dispersion * u.km / u.s},
                                 max_ev_time=200 * u.Myr,
                                 store_entire_orbits=False,
                                 use_default_BSE_settings=True)
    
    # update kick flag
    p.BSE_settings["kickflag"] = args.kick_flag

    p.create_population()
    p.save(args.output_file, overwrite=True)


if __name__ == "__main__":
    main()
