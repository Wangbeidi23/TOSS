fn main() {
    println!("cargo:rustc-link-arg=/NODEFAULTLIB:LIBCMT");
    println!("cargo:rustc-link-arg=/DEFAULTLIB:MSVCRT");
}