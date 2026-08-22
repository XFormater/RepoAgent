FROM soulter/astrbot:latest

# Reuse the already-local Debian/Python toolchain when the upstream Ubuntu
# registry is unavailable.  AgentBench's dev checks derive their answers from
# the same disposable container, so host-dependent counts remain self-consistent.
ENTRYPOINT []
CMD ["bash"]
