use ufo50ppo::games::ninpek::WINDOW_TITLE;
use ufo50ppo::platform::win32::input::{Input, VK_DOWN, VK_ESCAPE, VK_Z, vk_noop};

fn main() -> windows::core::Result<()> {
    let n: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let mut input = Input::new(WINDOW_TITLE)?;
    println!("Connected to '{}', running reset_game x{}", WINDOW_TITLE, n);
    println!("Watch the game window — each reset should return to gameplay.\n");

    for i in 1..=n {
        println!("[{}/{}] reset_game", i, n);
        input.reset_game(
            &[
                VK_ESCAPE,
                VK_DOWN,
                VK_Z,
                VK_Z,
                vk_noop(800),
                VK_Z,
                vk_noop(700),
                VK_Z,
            ],
            25,
        );
        std::thread::sleep(std::time::Duration::from_secs(3));
    }

    println!("\nDone. Releasing held keys.");
    input.release_all();
    Ok(())
}
