# Run inside `nix develop` - the flake wraps pixi with steam-run so its
# downloaded binaries (xgboost, etc.) can find ld-linux/libc on NixOS.

train-model:
    pixi run train-model

test-model:
    pixi run test-model

grid-model:
    pixi run grid-model
