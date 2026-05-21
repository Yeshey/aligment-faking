{ lib, ... }: {
  imports = [ ./vllm.nix ];

  perSystem = { pkgs, config, ... }: {
    devShells.vllm-cuda = pkgs.mkShell {
      name = "vllm-cuda";
      buildInputs = [ pkgs.python3 pkgs.python3Packages.pip pkgs.git pkgs.cudatoolkit ];
      shellHook = ''
        [ ! -d .venv ] && python -m venv .venv
        source .venv/bin/activate
        pip install -q vllm openai pandas tqdm
        export CUDA_HOME="${pkgs.cudatoolkit}"
        export LD_LIBRARY_PATH="${pkgs.cudatoolkit}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
        echo "=== vllm-cuda | VRAM: $VRAM ==="
        echo "Serve: vllm serve meta-llama/Meta-Llama-3-8B-Instruct"
      '';
    };
    # mkDefault: overrideable by leaf (vllm-cuda-l40 wins)
    devShells.default = lib.mkDefault config.devShells.vllm-cuda;
  };
}
