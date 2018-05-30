# NG, GNG adn FFNN for deformable objects monitoring

This is my replication of Cretu's system to learn to monitor the deformation of deformable materials.

## Instructions

### Segmentation
The image must be calibrated first with a GNG to select clusters. Run:

    python3 GNGFilterlessSegment.py <param_suits>

to generate a .pickle file that can be used.

Generate monitoring information from NG tracking
A file data/pickles/sponge_initial_contour.csv is needed to run the traking program.

    python3 GNGTrack.py sponge_set_1

