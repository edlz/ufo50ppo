// Embed libtorch's lib dir as an rpath so binaries run without DYLD_LIBRARY_PATH / LD_LIBRARY_PATH.
fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    let unix = std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix");
    if let (true, Ok(libtorch)) = (unix, std::env::var("LIBTORCH")) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libtorch}/lib");
    }
}
