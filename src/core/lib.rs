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

#![cfg_attr(ferrocene, warn(ferrocene::unvalidated))]

pub mod breaker;
pub mod error;
pub mod rate_limit;
pub mod log;
pub mod metrics;
pub mod telemetry;
pub mod gonets;
pub mod protocol;

pub use breaker::*;
pub use error::*;
pub use rate_limit::*;
pub use log::*;
pub use metrics::*;
pub use telemetry::*;
pub use gonets::*;
pub use protocol::*;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
