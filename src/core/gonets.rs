// MIT License
// Copyright (c) 2026 Person Detection Project
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use async_lock::Mutex;
use serialport::SerialPort;
use std::io::{Read, Write};
use crate::error::{BplaError, Result};

pub const MAX_AT_RESPONSE: usize = 512;
pub const AT_TIMEOUT_MS: u64 = 1_000;

pub struct GonetsModem {
    port: Mutex<Box<dyn SerialPort>>,
}

impl GonetsModem {
    pub fn open(port_name: &str, baud: u32) -> Result<Self> {
        let port: Box<dyn SerialPort> = serialport::new(port_name, baud)
            .timeout(std::time::Duration::from_millis(AT_TIMEOUT_MS))
            .open()
            .map_err(|_| BplaError::Transport("gonets open failed"))?;
        Ok(Self { port: Mutex::new(port) })
    }

    pub async fn at_command(&self, cmd: &[u8], buf: &mut [u8]) -> Result<usize> {
        let mut guard: async_lock::MutexGuard<'_, Box<dyn SerialPort + 'static>> = self.port.lock().await;
        Self::write_at(&mut *guard, cmd)?;
        Self::read_response(&mut *guard, buf)
    }

    pub async fn ping(&self) -> Result<bool> {
        let mut buf: [u8; MAX_AT_RESPONSE] = [0u8; MAX_AT_RESPONSE];
        let n: usize = self.at_command(b"AT", &mut buf).await?;
        Ok(buf[..n].windows(2).any(|w: &[u8]| w == b"OK"))
    }

    pub async fn send_telemetry(&self, data: &[u8]) -> Result<()> {
        let mut buf: [u8; MAX_AT_RESPONSE] = [0u8; MAX_AT_RESPONSE];
        let mut guard: async_lock::MutexGuard<'_, Box<dyn SerialPort + 'static>> = self.port.lock().await;

        Self::write_at(&mut *guard, b"AT+GONETS=SEND")?;
        Self::read_response(&mut *guard, &mut buf)?;

        guard.write_all(data).map_err(|_| BplaError::Transport("gonets telemetry write failed"))?;
        guard.flush().map_err(|_| BplaError::Transport("gonets telemetry flush failed"))?;
        Ok(())
    }

    fn write_at<P: Write + ?Sized>(port: &mut P, cmd: &[u8]) -> Result<()> {
        port.write_all(cmd).map_err(|_| BplaError::Transport("gonets write failed"))?;
        port.write_all(b"\r\n").map_err(|_| BplaError::Transport("gonets write failed"))?;
        port.flush().map_err(|_| BplaError::Transport("gonets flush failed"))
    }

    fn read_response<P: Read + ?Sized>(port: &mut P, buf: &mut [u8]) -> Result<usize> {
        match port.read(buf) {
            Ok(n) => Ok(n),
            Err(e) if is_no_response(&e) => Ok(0),
            Err(_) => Err(BplaError::Transport("gonets read failed")),
        }
    }
}

fn is_no_response(e: &std::io::Error) -> bool {
    matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock)
}