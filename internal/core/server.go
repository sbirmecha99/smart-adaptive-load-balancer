package core

import "time"

type Backend struct{
	Address string
	Weight int
	Alive bool
	ActiveConns int64
	Latency time.Duration
	ErrorCount int64
}