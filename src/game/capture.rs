use windows::{
    Foundation::TypedEventHandler, Graphics::Capture::*, Graphics::DirectX::Direct3D11::*,
    Graphics::DirectX::*, Win32::Foundation::*, Win32::Graphics::Direct3D::Fxc::D3DCompile,
    Win32::Graphics::Direct3D::*, Win32::Graphics::Direct3D11::*, Win32::Graphics::Dxgi::Common::*,
    Win32::Graphics::Dxgi::*,
    Win32::System::WinRT::Direct3D11::CreateDirect3D11DeviceFromDXGIDevice,
    Win32::System::WinRT::Direct3D11::IDirect3DDxgiInterfaceAccess,
    Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop,
    Win32::UI::WindowsAndMessaging::*, core::*,
};

/// Crop rectangle for extracting client area from a captured window frame.
#[derive(Clone, Copy)]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Find the title bar height by scanning down x=width/2 for the biggest color jump.
pub fn detect_border_height(bgra: &[u8], width: u32, height: u32) -> u32 {
    let scan_x = width / 2; // avoid potential UI elements on the left edge
    let search_range = height / 10;
    let mut max_diff = 0u32;
    let mut max_y = 0u32;
    for y in 1..search_range {
        let prev_i = ((y - 1) * width + scan_x) as usize * 4;
        let curr_i = (y * width + scan_x) as usize * 4;
        let diff = (bgra[prev_i] as i32 - bgra[curr_i] as i32).unsigned_abs()
            + (bgra[prev_i + 1] as i32 - bgra[curr_i + 1] as i32).unsigned_abs()
            + (bgra[prev_i + 2] as i32 - bgra[curr_i + 2] as i32).unsigned_abs();
        if diff > max_diff {
            max_diff = diff;
            max_y = y;
        }
    }
    max_y
}

fn create_d3d_device() -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    let mut device = None;
    let mut context = None;
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            None,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&[D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )?;
    }
    Ok((device.unwrap(), context.unwrap()))
}

pub fn init() -> Result<()> {
    use windows::Win32::System::WinRT::*;
    let options = DispatcherQueueOptions {
        dwSize: std::mem::size_of::<DispatcherQueueOptions>() as u32,
        threadType: DQTYPE_THREAD_CURRENT,
        apartmentType: DQTAT_COM_ASTA,
    };
    // Intentionally leaked — must outlive the thread's message pump
    let controller = unsafe { CreateDispatcherQueueController(options)? };
    std::mem::forget(controller);
    Ok(())
}

const HLSL: &str = r"
Texture2D src : register(t0);
SamplerState samp : register(s0);

struct VS_OUT { float4 pos : SV_Position; float2 uv : TEXCOORD; };

VS_OUT vs_main(uint id : SV_VertexID) {
    // fullscreen triangle from vertex ID
    VS_OUT o;
    o.uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}

float4 ps_main(VS_OUT i) : SV_Target {
    return src.Sample(samp, i.uv);
}
";

fn compile_shader(source: &str, entry: &str, target: &str) -> Result<ID3DBlob> {
    let source = source.as_bytes();
    let entry = std::ffi::CString::new(entry).unwrap();
    let target = std::ffi::CString::new(target).unwrap();
    let mut blob = None;
    unsafe {
        D3DCompile(
            source.as_ptr() as _,
            source.len(),
            None,
            None,
            None,
            PCSTR(entry.as_ptr() as _),
            PCSTR(target.as_ptr() as _),
            0,
            0,
            &mut blob,
            None,
        )?;
    }
    Ok(blob.unwrap())
}

/// Downsamples captured frames on the GPU, then reads back only a tiny texture.
pub struct FrameReader {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    vs: ID3D11VertexShader,
    ps: ID3D11PixelShader,
    sampler: ID3D11SamplerState,
    // Small render target + staging (created for a given output size)
    rt: Option<ID3D11Texture2D>,
    rtv: Option<ID3D11RenderTargetView>,
    staging: Option<ID3D11Texture2D>,
    // Cached crop texture (reused across frames)
    crop_tex: Option<ID3D11Texture2D>,
    crop_w: u32,
    crop_h: u32,
    out_w: u32,
    out_h: u32,
    buf: Vec<u8>,
}

impl FrameReader {
    fn new(device: ID3D11Device, context: ID3D11DeviceContext) -> Result<Self> {
        let vs_blob = compile_shader(HLSL, "vs_main", "vs_5_0")?;
        let ps_blob = compile_shader(HLSL, "ps_main", "ps_5_0")?;

        let vs = unsafe {
            let code = std::slice::from_raw_parts(
                vs_blob.GetBufferPointer() as *const u8,
                vs_blob.GetBufferSize(),
            );
            let mut vs = None;
            device.CreateVertexShader(code, None, Some(&mut vs))?;
            vs.unwrap()
        };

        let ps = unsafe {
            let code = std::slice::from_raw_parts(
                ps_blob.GetBufferPointer() as *const u8,
                ps_blob.GetBufferSize(),
            );
            let mut ps = None;
            device.CreatePixelShader(code, None, Some(&mut ps))?;
            ps.unwrap()
        };

        let sampler_desc = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT,
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
            ..Default::default()
        };
        let mut sampler = None;
        unsafe { device.CreateSamplerState(&sampler_desc, Some(&mut sampler))? };
        let sampler = sampler.unwrap();

        Ok(Self {
            device,
            context,
            vs,
            ps,
            sampler,
            rt: None,
            rtv: None,
            staging: None,
            crop_tex: None,
            crop_w: 0,
            crop_h: 0,
            out_w: 0,
            out_h: 0,
            buf: Vec::new(),
        })
    }

    fn ensure_targets(&mut self, w: u32, h: u32) -> Result<()> {
        if self.out_w == w && self.out_h == h && self.rt.is_some() {
            return Ok(());
        }
        let desc = D3D11_TEXTURE2D_DESC {
            Width: w,
            Height: h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut rt = None;
        unsafe { self.device.CreateTexture2D(&desc, None, Some(&mut rt))? };
        let rt = rt.unwrap();
        let mut rtv = None;
        unsafe {
            self.device
                .CreateRenderTargetView(&rt, None, Some(&mut rtv))?
        };
        let rtv = rtv.unwrap();

        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            ..desc
        };
        let mut staging = None;
        unsafe {
            self.device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging))?
        };

        self.rt = Some(rt);
        self.rtv = Some(rtv);
        self.staging = Some(staging.unwrap());
        self.out_w = w;
        self.out_h = h;
        Ok(())
    }

    /// Downsample the captured frame on the GPU to `(out_w, out_h)` and read back
    /// only that many pixels. Returns RGBA pixel data.
    pub fn read(
        &mut self,
        frame: &Direct3D11CaptureFrame,
        out_w: u32,
        out_h: u32,
    ) -> Result<&[u8]> {
        self.ensure_targets(out_w, out_h)?;

        let surface = frame.Surface()?;
        let access: IDirect3DDxgiInterfaceAccess = surface.cast()?;
        let texture: ID3D11Texture2D = unsafe { access.GetInterface()? };

        let mut srv = None;
        unsafe {
            self.device
                .CreateShaderResourceView(&texture, None, Some(&mut srv))?
        };
        let srv = srv.unwrap();

        let ctx = &self.context;
        let rtv = self.rtv.as_ref().unwrap();
        let staging = self.staging.as_ref().unwrap();

        unsafe {
            ctx.OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: out_w as f32,
                Height: out_h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.PSSetShader(&self.ps, None);
            ctx.PSSetShaderResources(0, Some(&[Some(srv)]));
            ctx.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            ctx.Draw(3, 0);

            // Unbind SRV so the texture can be released
            let empty: [Option<ID3D11ShaderResourceView>; 1] = [None];
            ctx.PSSetShaderResources(0, Some(&empty));

            ctx.CopyResource(staging, self.rt.as_ref().unwrap());
        }

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { ctx.Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))? };

        let w = out_w as usize;
        let h = out_h as usize;
        let row_pitch = mapped.RowPitch as usize;
        let src = unsafe { std::slice::from_raw_parts(mapped.pData as *const u8, row_pitch * h) };

        self.buf.clear();
        self.buf.reserve(w * h * 4);
        for y in 0..h {
            let row = &src[y * row_pitch..y * row_pitch + w * 4];
            self.buf.extend_from_slice(row);
        }

        unsafe { ctx.Unmap(staging, 0) };

        // Data is BGRA from the GPU; callers use it directly
        Ok(&self.buf)
    }

    /// Read raw pixels from the captured frame without any GPU processing.
    pub fn read_raw(&mut self, frame: &Direct3D11CaptureFrame) -> Result<(u32, u32, Vec<u8>)> {
        let surface = frame.Surface()?;
        let access: IDirect3DDxgiInterfaceAccess = surface.cast()?;
        let texture: ID3D11Texture2D = unsafe { access.GetInterface()? };
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { texture.GetDesc(&mut desc) };

        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            ..desc
        };
        let mut staging = None;
        unsafe {
            self.device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging))?
        };
        let staging = staging.unwrap();
        unsafe { self.context.CopyResource(&staging, &texture) };

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?
        };

        let w = desc.Width as usize;
        let h = desc.Height as usize;
        let row_pitch = mapped.RowPitch as usize;
        let raw = unsafe { std::slice::from_raw_parts(mapped.pData as *const u8, row_pitch * h) };

        let mut pixels = vec![0u8; w * h * 4];
        for y in 0..h {
            let src = &raw[y * row_pitch..y * row_pitch + w * 4];
            pixels[y * w * 4..(y + 1) * w * 4].copy_from_slice(src);
        }
        unsafe { self.context.Unmap(&staging, 0) };

        Ok((desc.Width, desc.Height, pixels))
    }

    /// Like `read`, but first crops the captured frame to the given rect.
    /// Use with `client_crop_rect()` to exclude window borders.
    pub fn read_cropped(
        &mut self,
        frame: &Direct3D11CaptureFrame,
        crop: &CropRect,
        out_w: u32,
        out_h: u32,
    ) -> Result<&[u8]> {
        self.ensure_targets(out_w, out_h)?;

        let surface = frame.Surface()?;
        let access: IDirect3DDxgiInterfaceAccess = surface.cast()?;
        let texture: ID3D11Texture2D = unsafe { access.GetInterface()? };

        // Cache crop texture across frames
        if self.crop_w != crop.w || self.crop_h != crop.h || self.crop_tex.is_none() {
            let crop_desc = D3D11_TEXTURE2D_DESC {
                Width: crop.w,
                Height: crop.h,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let mut tex = None;
            unsafe {
                self.device
                    .CreateTexture2D(&crop_desc, None, Some(&mut tex))?
            };
            self.crop_tex = Some(tex.unwrap());
            self.crop_w = crop.w;
            self.crop_h = crop.h;
        }
        let crop_tex = self.crop_tex.as_ref().unwrap();

        let src_box = D3D11_BOX {
            left: crop.x,
            top: crop.y,
            front: 0,
            right: crop.x + crop.w,
            bottom: crop.y + crop.h,
            back: 1,
        };
        unsafe {
            self.context
                .CopySubresourceRegion(crop_tex, 0, 0, 0, 0, &texture, 0, Some(&src_box));
        }

        let mut srv = None;
        unsafe {
            self.device
                .CreateShaderResourceView(crop_tex, None, Some(&mut srv))?
        };
        let srv = srv.unwrap();

        let ctx = &self.context;
        let rtv = self.rtv.as_ref().unwrap();
        let staging = self.staging.as_ref().unwrap();

        unsafe {
            ctx.OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: out_w as f32,
                Height: out_h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.PSSetShader(&self.ps, None);
            ctx.PSSetShaderResources(0, Some(&[Some(srv)]));
            ctx.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            ctx.Draw(3, 0);

            let empty: [Option<ID3D11ShaderResourceView>; 1] = [None];
            ctx.PSSetShaderResources(0, Some(&empty));
            ctx.CopyResource(staging, self.rt.as_ref().unwrap());
        }

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { ctx.Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))? };

        let w = out_w as usize;
        let h = out_h as usize;
        let row_pitch = mapped.RowPitch as usize;
        let src = unsafe { std::slice::from_raw_parts(mapped.pData as *const u8, row_pitch * h) };

        self.buf.clear();
        self.buf.reserve(w * h * 4);
        for y in 0..h {
            let row = &src[y * row_pitch..y * row_pitch + w * 4];
            self.buf.extend_from_slice(row);
        }

        unsafe { ctx.Unmap(staging, 0) };
        Ok(&self.buf)
    }

    /// Save current buffer as a BMP file for debugging.
    pub fn save_debug_bmp(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let w = self.out_w;
        let h = self.out_h;
        let row_bytes = (w * 3 + 3) & !3; // BMP rows are 4-byte aligned
        let pixel_size = row_bytes * h;
        let file_size = 54 + pixel_size;

        let mut f = std::fs::File::create(path)?;
        // BMP header
        f.write_all(b"BM")?;
        f.write_all(&(file_size).to_le_bytes())?;
        f.write_all(&0u32.to_le_bytes())?; // reserved
        f.write_all(&54u32.to_le_bytes())?; // pixel data offset
        // DIB header
        f.write_all(&40u32.to_le_bytes())?; // header size
        f.write_all(&(w as i32).to_le_bytes())?;
        f.write_all(&(-(h as i32)).to_le_bytes())?; // negative = top-down
        f.write_all(&1u16.to_le_bytes())?; // planes
        f.write_all(&24u16.to_le_bytes())?; // bpp
        f.write_all(&0u32.to_le_bytes())?; // no compression
        f.write_all(&pixel_size.to_le_bytes())?;
        f.write_all(&[0u8; 16])?; // rest of DIB header

        let mut row = vec![0u8; row_bytes as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let src = y * w as usize * 4 + x * 4;
                let dst = x * 3;
                row[dst] = self.buf[src]; // B
                row[dst + 1] = self.buf[src + 1]; // G
                row[dst + 2] = self.buf[src + 2]; // R
            }
            f.write_all(&row)?;
        }
        Ok(())
    }
}

/// Captures frames from the given window. The callback receives the crop rect,
/// frame, and reader. Use `reader.read_cropped(frame, crop, w, h)` to get
/// client-area-only pixels. Return `true` to continue, `false` to stop.
pub fn run<F>(window_title: &str, on_frame: F) -> Result<()>
where
    F: FnMut(&CropRect, &Direct3D11CaptureFrame, &mut FrameReader) -> bool + Send + 'static,
{
    let title = format!("{}\0", window_title);
    let hwnd = unsafe { FindWindowA(None, PCSTR(title.as_ptr()))? };

    let interop = windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()?;
    let item: GraphicsCaptureItem = unsafe { interop.CreateForWindow(hwnd)? };

    let (device, context) = create_d3d_device()?;

    let dxgi_device: IDXGIDevice = device.cast()?;
    let inspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi_device)? };
    let d3d_device: IDirect3DDevice = inspectable.cast()?;

    let size = item.Size()?;
    let pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        &d3d_device,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        2,
        size,
    )?;

    let session = pool.CreateCaptureSession(&item)?;

    let mut reader = FrameReader::new(device, context)?;
    let mut on_frame = on_frame;
    let main_thread_id = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
    let mut crop: Option<CropRect> = None;

    pool.FrameArrived(&TypedEventHandler::new(
        move |pool: &Option<Direct3D11CaptureFramePool>, _| {
            let pool = pool.as_ref().unwrap();
            let frame = pool.TryGetNextFrame()?;

            // Detect border on first frame by reading raw captured pixels
            if crop.is_none() {
                match reader.read_raw(&frame) {
                    Ok((full_w, full_h, pixels)) => {
                        let border_h = detect_border_height(&pixels, full_w, full_h);
                        crop = Some(CropRect {
                            x: 0,
                            y: border_h,
                            w: full_w,
                            h: full_h - border_h,
                        });
                        println!(
                            "Detected border: {}px, client area: {}x{}",
                            border_h,
                            full_w,
                            full_h - border_h
                        );
                    }
                    Err(e) => {
                        eprintln!("Border detection failed: {}, using no crop", e);
                        crop = Some(CropRect {
                            x: 0,
                            y: 0,
                            w: 1,
                            h: 1,
                        });
                    }
                }
                frame.Close()?;
                return Ok(());
            }

            let keep_going = on_frame(crop.as_ref().unwrap(), &frame, &mut reader);

            frame.Close()?;

            if !keep_going {
                unsafe {
                    PostThreadMessageA(main_thread_id, WM_QUIT, WPARAM(0), LPARAM(0))?;
                }
            }

            Ok(())
        },
    ))?;

    session.SetIsBorderRequired(false).ok();
    session.StartCapture()?;

    unsafe {
        let mut msg = MSG::default();
        while GetMessageA(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
    }

    Ok(())
}
