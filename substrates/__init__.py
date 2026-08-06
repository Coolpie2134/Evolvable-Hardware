"""
substrates - the four grown-circuit substrates evolved by this project.

Each is a self-contained backend (its own genome, growth, dynamics, targets and
GA); they share only problem definitions and trace-scoring maths:

  * substrates.snn      - spiking LIF neuron array (was snn_evo)
  * substrates.nervous  - Edwards EH'02 honeycomb "nervous net" (was nv_evo)
  * substrates.fnv      - Functional NV Net: fixed physical components on
                          directed honeycomb wires
  * substrates.lut      - paper Architecture 2 square LUT array (was lut_evo)
"""
