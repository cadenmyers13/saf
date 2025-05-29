# Skyrmion Lattices

A, possibly ML enabled, engine for rapidly returning the orientation of the underlying skyrmion lattice from a stack of SANS images

We have developed a collaboration with Morten Eskildsen at U. Notre Dame with a goal of developing AI-enabled tools for extracting the orientation of skyrmion lattices from noisy and inhomogeneous SANS data. Many hundreds of SANS 2D scattering images are generated from time-series experiments where the skyrmion lattice is being driven by radially symmetric electrical or thermal currents. The goal is to develop ML models that can take the images as inputs and output key physical parameters such as the orientation of the lattice, which rotates under driving.

## High Level Goal

1. Develop algorithm that takes SANS images and outputs angular velocity
2. Release algorithm as package
3. Contribute to a paper with Nathan

## Team

- Caden Myers (cadenmyers13)
- Yucong Chen (yucongalicechen)
- Andrew Yang (Sparks29032)
- Nathan Chalus (nchalus1)

## Summer 2024

The summer with be considered a success if we:

- Complete high level goals 1 and 2
- Fairly deep into the manuscript writing

## Milestones

### By 2024-06-14

- [x] Discuss details of the project
- [x] Identify ML-based approaches for goal 1 (CRNN?)
- [x] Come up with ideas to handle noise (symmetry of the system?)
- [x] Meet with Savannah from DSI

### By 2024-06-28

- [ ] Modify CNN from space group mining paper (Liu, et al. 2019)
- [ ] Train and test CNN on small data subset
- [x] Meet with Savannah from DSI

## Models

- <u>**CNN from space group mining paper (Liu, et al. 2019)**
- Rigid Hexagonal symmetry
- Slack Hexagonal symmetry
- e3NN
- EGNN
- unsupervised vs. supervised learning

## Future Direction

- Create real space image using angular velocity

## Collaboration

The SANS data used in this project is courtesy of the Morten Eskildsen Group at the University of Notre Dame
