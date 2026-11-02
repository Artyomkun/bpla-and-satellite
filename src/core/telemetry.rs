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

use tcslog::{Format, LogWrite};
use crate::error::{BplaError, Result};

pub struct TelemetryLog {
    writer: LogWrite,
}

impl TelemetryLog {
    pub fn open(dir: &str, segment_size: u32) -> Result<Self> {
        let writer: LogWrite = LogWrite::new(dir, "seg-", ".tcslog", segment_size, Format::VariableTsRc, ()).map_err(|_| BplaError::Telemetry("open failed"))?;
        Ok(Self { writer })
    }

    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write(data).map_err(|_| BplaError::Telemetry("write failed"))?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().map_err(|_| BplaError::Telemetry("flush failed"))
    }
}