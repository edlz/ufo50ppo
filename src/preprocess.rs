use tch::{Device, Tensor, Kind};

const STACK_SIZE: i64 = 4;
const FRAME_SIZE: i64 = 84;

pub struct FrameStack {
    buffer: Tensor, // [1, 4, 84, 84] on device
    count: usize,
}

impl FrameStack {
    pub fn new(device: Device) -> Self {
        Self {
            buffer: Tensor::zeros([1, STACK_SIZE, FRAME_SIZE, FRAME_SIZE], (Kind::Float, device)),
            count: 0,
        }
    }

    /// Takes BGRA pixels (from FrameReader, already downscaled to 84x84),
    /// converts to grayscale, normalizes, pushes into the frame stack.
    /// Returns a clone of the stacked observation [1, 4, 84, 84].
    pub fn push(&mut self, bgra: &[u8], width: u32, height: u32) -> Tensor {
        let rgba = bgra;
        let w = width as usize;
        let h = height as usize;

        // BGRA → grayscale f32 normalized to [0, 1]
        let mut gray = vec![0f32; w * h];
        for i in 0..w * h {
            let b = rgba[i * 4] as f32;
            let g = rgba[i * 4 + 1] as f32;
            let r = rgba[i * 4 + 2] as f32;
            gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        }

        let frame = Tensor::from_slice(&gray)
            .reshape([1, 1, h as i64, w as i64])
            .to_device(self.buffer.device());

        // Resize to 84x84 if not already that size
        let frame = if h as i64 != FRAME_SIZE || w as i64 != FRAME_SIZE {
            frame.upsample_bilinear2d([FRAME_SIZE, FRAME_SIZE], false, None, None)
        } else {
            frame
        };

        if self.count == 0 {
            // Fill all 4 channels with the first frame
            for i in 0..STACK_SIZE {
                self.buffer.narrow(1, i, 1).copy_(&frame);
            }
        } else {
            // Shift channels 0..2 ← 1..3, write new frame into channel 3
            let shifted = self.buffer.narrow(1, 1, STACK_SIZE - 1).shallow_clone();
            self.buffer.narrow(1, 0, STACK_SIZE - 1).copy_(&shifted);
            self.buffer.narrow(1, STACK_SIZE - 1, 1).copy_(&frame);
        }
        self.count += 1;

        self.buffer.shallow_clone()
    }

    pub fn reset(&mut self) {
        self.buffer.zero_();
        self.count = 0;
    }
}
