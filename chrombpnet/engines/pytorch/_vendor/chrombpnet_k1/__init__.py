"""The ChromBPNet optimization kit's Triton forward kernels, vendored.

Deliberately does NOT import `kernels`: that module imports triton, and
`import chrombpnet` must stay usable on a machine without it. The local
wrapper `chrombpnet.engines.pytorch.triton_forward` imports it only once the route has been
checked.
"""
