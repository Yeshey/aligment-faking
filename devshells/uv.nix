{ ... }: {
  perSystem = { pkgs, lib, config, ... }: {
    devShells.uv = pkgs.mkShell {
      name = "uv";
      buildInputs = [ pkgs.uv pkgs.python312 pkgs.git pkgs.stdenv.cc.cc.lib ];
      shellHook = ''
        export UV_PYTHON="${pkgs.python312}/bin/python"
        export UV_PROJECT_ENVIRONMENT=".venv"
        export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "Downloading dependencies, might take a long time on first run..."
        uv sync --frozen 2>/dev/null || uv sync
        source .venv/bin/activate
        VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "no GPU")
        echo "=== uv | VRAM: $VRAM ==="
      '';
    };
    devShells.default = config.devShells.uv;
  };
}