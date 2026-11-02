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

use breaker::{CircuitBreaker, CircuitBreakerConfig};
use crate::error::{BplaError, Result};

pub struct Breakers {
    http: CircuitBreaker,
    udp: CircuitBreaker,
}

impl Breakers {
    pub fn new() -> Self {
        Self {
            http: CircuitBreaker::new(CircuitBreakerConfig::standard()),
            udp: CircuitBreaker::new(CircuitBreakerConfig::fast_fail()),
        }
    }

    pub async fn call_http<F, Fut, T>(&self, f: F) -> Result<T> where F: FnOnce() -> Fut, Fut: std::future::Future<Output = Result<T>> {
        match self.http.call(f).await {
            Ok(v) => Ok(v),
            Err(_) => Err(BplaError::CircuitOpen("http")),
        }
    }

    pub async fn call_udp<F, Fut, T>(&self, f: F) -> Result<T> where F: FnOnce() -> Fut, Fut: std::future::Future<Output = Result<T>> {
        match self.udp.call(f).await {
            Ok(value) => Ok(value),
            Err(_) => Err(BplaError::CircuitOpen("udp")),
        }
    }
}

impl Default for Breakers {
    fn default() -> Self {
        Self::new()
    }
}