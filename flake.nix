{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [ pkgs.jetbrains.rider ];

        shellHook = ''
          LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${
            (
              with pkgs;
              lib.makeLibraryPath [
                glibc # libdl
                gtk3 # libglib-2.0.so.0 libgobject-2.0.so.0 libgtk-3.so.0 libgdk-3.so.0
                libGL # libGL.so.1
                xorg.libICE # libICE.so.6
                xorg.libSM # libSM.so.6
                xorg.libX11 # libX11 libX11.so.6
                xorg.libXcursor # libXcursor.so.1
                xorg.libXi # libXi.so.6
                xorg.libXrandr # libXrandr.so.2
                fontconfig # libfontconfig.so.1
              ]
            )
          }"
        '';
      };
    };
}
