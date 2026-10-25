use tch::{Device, Kind, Tensor};

const STACK_SIZE: i64 = 8;
const COLORS: i64 = 3;
/// Model input channels: RGB for each stacked frame.
pub const CHANNELS: i64 = STACK_SIZE * COLORS;

/// Stacks the last STACK_SIZE frames as uint8 RGB; the model scales to [0, 1] itself, which
/// keeps rollout buffers at a quarter of the float size.
pub struct FrameStack {
    buffer: Tensor,
    width: i64,
    height: i64,
    count: usize,
}

impl FrameStack {
    pub fn new(device: Device, width: u32, height: u32) -> Self {
        let (w, h) = (width as i64, height as i64);
        Self {
            buffer: Tensor::zeros([1, CHANNELS, h, w], (Kind::Uint8, device)),
            width: w,
            height: h,
            count: 0,
        }
    }

    /// `bgra`: one frame at the stack's size.
    pub fn push(&mut self, bgra: &[u8]) -> Tensor {
        let n = (self.width * self.height) as usize;
        let mut planar = vec![0u8; n * 3];
        for (i, px) in bgra.chunks_exact(4).take(n).enumerate() {
            planar[i] = px[2];
            planar[n + i] = px[1];
            planar[2 * n + i] = px[0];
        }
        let frame = Tensor::from_slice(&planar)
            .reshape([1, COLORS, self.height, self.width])
            .to_device(self.buffer.device());

        if self.count == 0 {
            for i in 0..STACK_SIZE {
                self.buffer.narrow(1, i * COLORS, COLORS).copy_(&frame);
            }
        } else {
            let shifted = self.buffer.narrow(1, COLORS, CHANNELS - COLORS).copy();
            self.buffer.narrow(1, 0, CHANNELS - COLORS).copy_(&shifted);
            self.buffer
                .narrow(1, CHANNELS - COLORS, COLORS)
                .copy_(&frame);
        }
        self.count += 1;

        self.buffer.copy()
    }

    pub fn reset(&mut self) {
        let _ = self.buffer.zero_();
        self.count = 0;
    }
}
