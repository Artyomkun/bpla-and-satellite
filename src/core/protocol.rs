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

use crate::error::{BplaError, Result};

pub const MAX_PAYLOAD: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MsgType {
    Telemetry = 0x01,
    Command   = 0x02,
    Ack       = 0x03,
    Nack      = 0x04,
}

impl MsgType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Telemetry),
            0x02 => Some(Self::Command),
            0x03 => Some(Self::Ack),
            0x04 => Some(Self::Nack),
            _ => None
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Header {
    pub msg_type: u8,
    pub drone_id: u32,
    pub seq: u16,
    pub payload_len: u16
}

impl Header {
    pub const SIZE: usize = 9;

    pub fn new(msg_type: MsgType, drone_id: u32, seq: u16, payload_len: u16) -> Self {
        Self {msg_type: msg_type as u8, drone_id, seq, payload_len}
    }

    pub fn encode(&self, buf: &mut [u8]) -> Result<usize> {
        if buf.len() < Self::SIZE {return Err(BplaError::Transport("header buffer too small"));}
        buf[0] = self.msg_type;
        buf[1..5].copy_from_slice(&self.drone_id.to_be_bytes());
        buf[5..7].copy_from_slice(&self.seq.to_be_bytes());
        buf[7..9].copy_from_slice(&self.payload_len.to_be_bytes());
        Ok(Self::SIZE)
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {return Err(BplaError::Transport("header buffer too small"));}
        Ok(Self {
            msg_type: buf[0],
            drone_id: u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]),
            seq: u16::from_be_bytes([buf[5], buf[6]]),
            payload_len: u16::from_be_bytes([buf[7], buf[8]])
        })
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TelemetryPayload {
    pub lat: f64,
    pub lon: f64,
    pub alt_m: f32,
    pub battery_pct: u8,
    pub timestamp: u64
}

impl TelemetryPayload {
    pub const SIZE: usize = 8 + 8 + 4 + 1 + 8;

    pub fn encode(&self, buf: &mut [u8]) -> Result<usize> {
        if buf.len() < Self::SIZE {return Err(BplaError::Transport("telemetry buffer too small"));}
        buf[0..8].copy_from_slice(&self.lat.to_be_bytes());
        buf[8..16].copy_from_slice(&self.lon.to_be_bytes());
        buf[16..20].copy_from_slice(&self.alt_m.to_be_bytes());
        buf[20] = self.battery_pct;
        buf[21..29].copy_from_slice(&self.timestamp.to_be_bytes());
        Ok(Self::SIZE)
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {return Err(BplaError::Transport("telemetry buffer too small"));}
        Ok(Self {
            lat: f64::from_be_bytes([buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]]),
            lon: f64::from_be_bytes([buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]]),
            alt_m: f32::from_be_bytes([buf[16], buf[17], buf[18], buf[19]]),
            battery_pct: buf[20],
            timestamp: u64::from_be_bytes([buf[21], buf[22], buf[23], buf[24], buf[25], buf[26], buf[27], buf[28]])
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CommandCode {
    ReturnToBase = 0x01,
    HoldPosition = 0x02,
    ResumeMission = 0x03,
}

impl CommandCode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::ReturnToBase),
            0x02 => Some(Self::HoldPosition),
            0x03 => Some(Self::ResumeMission),
            _ => None
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CommandPayload {
    pub command: u8,
    pub param: f32
}

impl CommandPayload {
    pub const SIZE: usize = 1 + 4;

    pub fn encode(&self, buf: &mut [u8]) -> Result<usize> {
        if buf.len() < Self::SIZE {return Err(BplaError::Transport("command buffer too small"));}
        buf[0] = self.command;
        buf[1..5].copy_from_slice(&self.param.to_be_bytes());
        Ok(Self::SIZE)
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {return Err(BplaError::Transport("command buffer too small"));}
        Ok(Self {command: buf[0], param: f32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]])})
    }
}

pub struct Message {
    pub header: Header,
    pub payload: heapless::Vec<u8, MAX_PAYLOAD>
}

impl Message {
    pub fn telemetry(drone_id: u32, seq: u16, payload: &TelemetryPayload) -> Result<Self> {
        let mut payload_buf: [u8; 29] = [0u8; TelemetryPayload::SIZE];
        payload.encode(&mut payload_buf)?;
        let mut v = heapless::Vec::new();
        v.extend_from_slice(&payload_buf).map_err(|_| BplaError::Transport("payload too large"))?;
        Ok(Self {header: Header::new(MsgType::Telemetry, drone_id, seq, TelemetryPayload::SIZE as u16),payload: v})
    }

    pub fn command(drone_id: u32, seq: u16, payload: &CommandPayload) -> Result<Self> {
        let mut payload_buf: [u8; 5] = [0u8; CommandPayload::SIZE];
        payload.encode(&mut payload_buf)?;

        let mut v = heapless::Vec::new();
        v.extend_from_slice(&payload_buf).map_err(|_| BplaError::Transport("payload too large"))?;

        Ok(Self {header: Header::new(MsgType::Command, drone_id, seq, CommandPayload::SIZE as u16), payload: v})
    }

    pub fn encode(&self, buf: &mut [u8]) -> Result<usize> {
        let total: usize = Header::SIZE + self.payload.len();
        if buf.len() < total {return Err(BplaError::Transport("message buffer too small"));}
        let n: usize = self.header.encode(buf)?;
        buf[n..n + self.payload.len()].copy_from_slice(&self.payload);
        Ok(total)
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        let header: Header = Header::decode(buf)?;
        let start: usize = Header::SIZE;
        let end: usize = start + header.payload_len as usize;
        if buf.len() < end {return Err(BplaError::Transport("incomplete message"));}
        let mut payload = heapless::Vec::new();
        payload.extend_from_slice(&buf[start..end]).map_err(|_| BplaError::Transport("payload too large"))?;
        Ok(Self { header, payload })
    }
}