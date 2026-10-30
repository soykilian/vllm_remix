# vLLM-REMIX

A fork of [vLLM](https://github.com/vllm-project/vllm) that adds a heterogeneous-device partitioning framework called **REMIX** on top of vLLM's `torch.compile`-based piecewise compilation backend. REMIX lets you split a single LLM graph across multiple device types — GPU, attention accelerators, PIM, custom NPUs — and route each subgraph to the backend that runs it most efficiently.

### **Disclaimer** vLLM-REMIX is in the process of obtaining internal clearance for open sourcing. Expected date of publication of the code is 10th of November 20206.
