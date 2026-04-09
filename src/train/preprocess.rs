use tch::{Device, Kind, Tensor};

const STACK_SIZE: i64 = 4;

pub struct FrameStack {
    buffer: Tensor,
    frame_size: i64,
    count: usize,
}

impl FrameStack {
    pub fn new(device: Device, frame_size: u32) -> Self {
        let fs = frame_size as i64;
        Self {
            buffer: Tensor::zeros([1, STACK_SIZE, fs, fs], (Kind::Float, device)),
            frame_size: fs,
            count: 0,
        }
    }

    pub fn push(&mut self, bgra: &[u8], width: u32, height: u32) -> Tensor {
        let w = width as usize;
        let h = height as usize;

        // BT.601 luma coefficients.
        let mut gray = vec![0f32; w * h];
        for i in 0..w * h {
            let b = bgra[i * 4] as f32;
            let g = bgra[i * 4 + 1] as f32;
            let r = bgra[i * 4 + 2] as f32;
            gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        }

        let frame = Tensor::from_slice(&gray)
            .reshape([1, 1, h as i64, w as i64])
            .to_device(self.buffer.device());

        let frame = if h as i64 != self.frame_size || w as i64 != self.frame_size {
            frame.upsample_bilinear2d([self.frame_size, self.frame_size], false, None, None)
        } else {
            frame
        };

        if self.count == 0 {
            for i in 0..STACK_SIZE {
                self.buffer.narrow(1, i, 1).copy_(&frame);
            }
        } else {
            let shifted = self.buffer.narrow(1, 1, STACK_SIZE - 1).copy();
            self.buffer.narrow(1, 0, STACK_SIZE - 1).copy_(&shifted);
            self.buffer.narrow(1, STACK_SIZE - 1, 1).copy_(&frame);
        }
        self.count += 1;

        self.buffer.copy()
    }

    pub fn reset(&mut self) {
        let _ = self.buffer.zero_();
        self.count = 0;
    }
}
