{ inputs, ... }: {
  perSystem = { lib, system, config, ... }:
  let
    pkgs = import inputs.nixpkgs {
      inherit system;
      config.allowUnfree = true;
      config.cudaSupport = true;
    };
  in {
    devShells.uv = pkgs.mkShell {
      name = "uv";
      buildInputs = with pkgs; [
        uv
        python312
        git
        stdenv.cc.cc.lib
        zlib
        cudaPackages.cudatoolkit
        cudaPackages.cuda_cudart
        linuxPackages.nvidiaPackages.stable
      ];
      shellHook = ''
        export UV_PYTHON="${pkgs.python312}/bin/python"
        export UV_PROJECT_ENVIRONMENT=".venv"
        export LD_LIBRARY_PATH="/run/opengl-driver/lib:${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib pkgs.cudaPackages.cudatoolkit ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
        export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
        export PATH="${pkgs.glibc.bin}/bin:$PATH"
        mkdir -p /sbin 2>/dev/null || sudo mkdir -p /sbin 2>/dev/null || true
        sudo ln -sf "${pkgs.glibc.bin}/bin/ldconfig" /sbin/ldconfig 2>/dev/null || true
        sudo ldconfig 2>/dev/null || true
        uv sync --frozen 2>/dev/null || uv sync
        source .venv/bin/activate
        VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "no GPU")
        echo "=== uv+CUDA | VRAM: $VRAM ==="
      '';
    };

    devShells.default = config.devShells.uv;
  };
}