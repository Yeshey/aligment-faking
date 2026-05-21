{
  description = "Alignment Faking Replication";

  inputs = {
    nixpkgs.url     = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    import-tree.url = "github:vic/import-tree";
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        (inputs.import-tree ./modules)
        # Devshell hierarchy: change leaf to switch profile
        ./devshells/vllm-cuda-l40.nix   # L40 cluster (default)
        # ./devshells/vllm-cuda.nix     # any CUDA machine
        # ./devshells/vllm.nix          # CPU only
        # ./devshells/bnb.nix           # QLoRA fine-tuning
      ];
      systems = [ "x86_64-linux" "aarch64-linux" ];
    };
}
