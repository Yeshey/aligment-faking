{ ... }: {
  imports = [ ./uv.nix ];

  perSystem = { pkgs, lib, config, ... }: {
    devShells.uv-cuda = pkgs.mkShell {
      name = "uv-cuda";
      buildInputs = [
        pkgs.uv pkgs.python3 pkgs.git
        pkgs.cudatoolkit
        pkgs.cudaPackages.cudnn
        pkgs.stdenv.cc.cc.lib
      ];
      shellHook = ''
        export CUDA_HOME="${pkgs.cudatoolkit}"
        export LD_LIBRARY_PATH="${pkgs.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export UV_PYTHON="${pkgs.python3}/bin/python"
        export UV_PROJECT_ENVIRONMENT=".venv"
        uv sync --frozen 2>/dev/null || uv sync
        source .venv/bin/activate
        VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
        echo "=== uv-cuda | VRAM: $VRAM ==="
        echo "Run: python src/gen.py && python src/eval.py"
      '';
    };
    # priority 100 beats uv.nix's mkDefault (1000) — wins when both imported
    devShells.default = config.devShells.uv-cuda;
  };
}
