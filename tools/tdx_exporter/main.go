package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/injoyai/tdx"
	"github.com/injoyai/tdx/protocol"
)

type ETF struct {
	Symbol   string
	Name     string
	Category string
	Bucket   string
}

type DailyRow struct {
	Date      string  `json:"date"`
	Symbol    string  `json:"symbol"`
	TDXSymbol string  `json:"tdx_symbol"`
	Name      string  `json:"name"`
	Category  string  `json:"category"`
	Bucket    string  `json:"bucket"`
	Open      float64 `json:"open"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Close     float64 `json:"close"`
	Volume    int64   `json:"volume"`
	Amount    float64 `json:"amount"`
}

type GbbqRow struct {
	Date         string  `json:"date"`
	Symbol       string  `json:"symbol"`
	TDXSymbol    string  `json:"tdx_symbol"`
	CategoryCode int     `json:"category_code"`
	C1           float64 `json:"c1"`
	C2           float64 `json:"c2"`
	C3           float64 `json:"c3"`
	C4           float64 `json:"c4"`
}

var universe = []ETF{
	{"510050", "上证50ETF", "broad", "large_value"},
	{"510300", "沪深300ETF", "broad", "csi300"},
	{"510500", "中证500ETF", "broad", "csi500"},
	{"512100", "中证1000ETF", "broad", "csi1000"},
	{"159915", "创业板ETF", "broad", "chinext"},
	{"588000", "科创50ETF", "broad", "star50"},
	{"510880", "红利ETF", "broad", "dividend"},
	{"512800", "银行ETF", "industry", "bank"},
	{"512880", "证券ETF", "industry", "broker"},
	{"512760", "半导体ETF", "industry", "semiconductor"},
	{"512660", "军工ETF", "industry", "defense"},
	{"512010", "医药ETF", "industry", "pharma"},
	{"159928", "消费ETF", "industry", "consumer"},
	{"512400", "有色金属ETF", "industry", "nonferrous"},
	{"515220", "煤炭ETF", "industry", "coal"},
	{"515790", "光伏ETF", "industry", "solar"},
	{"515030", "新能源车ETF", "industry", "new_energy_vehicle"},
}

func main() {
	outDir := flag.String("out", filepath.Join("data", "staging"), "output staging directory")
	start := flag.String("start", "2010-01-01", "inclusive start date")
	universePath := flag.String("universe", "", "optional JSON universe file")
	flag.Parse()

	startDate, err := time.Parse(time.DateOnly, *start)
	if err != nil {
		fatalf("invalid -start: %v", err)
	}

	if err := os.MkdirAll(*outDir, 0755); err != nil {
		fatalf("create output directory: %v", err)
	}
	exportUniverse := universe
	if *universePath != "" {
		exportUniverse, err = loadUniverse(*universePath)
		if err != nil {
			fatalf("load universe: %v", err)
		}
	}

	bfqFile := mustCreate(filepath.Join(*outDir, "tdx_daily_bfq.jsonl"))
	defer bfqFile.Close()
	qfqFile := mustCreate(filepath.Join(*outDir, "tdx_daily_qfq_latest.jsonl"))
	defer qfqFile.Close()
	gbbqFile := mustCreate(filepath.Join(*outDir, "tdx_gbbq_events.jsonl"))
	defer gbbqFile.Close()

	bfqEnc := json.NewEncoder(bfqFile)
	qfqEnc := json.NewEncoder(qfqFile)
	gbbqEnc := json.NewEncoder(gbbqFile)

	client, err := tdx.DialDefault(tdx.WithRedial())
	if err != nil {
		fatalf("dial TDX: %v", err)
	}
	defer client.Close()

	for _, etf := range exportUniverse {
		if err := exportETF(client, etf, startDate, bfqEnc, qfqEnc, gbbqEnc); err != nil {
			fatalf("export %s %s: %v", etf.Symbol, etf.Name, err)
		}
		fmt.Fprintf(os.Stderr, "exported %s %s\n", etf.Symbol, etf.Name)
	}
}

func loadUniverse(path string) ([]ETF, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var items []ETF
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("empty universe")
	}
	seen := make(map[string]bool, len(items))
	for _, item := range items {
		if item.Symbol == "" || item.Name == "" || item.Category == "" || item.Bucket == "" {
			return nil, fmt.Errorf("incomplete universe item: %+v", item)
		}
		if seen[item.Symbol] {
			return nil, fmt.Errorf("duplicate symbol: %s", item.Symbol)
		}
		seen[item.Symbol] = true
	}
	return items, nil
}

func exportETF(client *tdx.Client, etf ETF, startDate time.Time, bfqEnc, qfqEnc, gbbqEnc *json.Encoder) error {
	code := tdxSymbol(etf.Symbol)

	resp, err := client.GetKlineDayAll(code)
	if err != nil {
		return fmt.Errorf("get day kline: %w", err)
	}
	if resp == nil || len(resp.List) == 0 {
		return fmt.Errorf("empty day kline")
	}

	gbbqResp, err := client.GetGbbq(code)
	if err != nil {
		return fmt.Errorf("get gbbq: %w", err)
	}
	if gbbqResp == nil {
		return fmt.Errorf("nil gbbq response")
	}

	for _, event := range gbbqResp.List {
		if event == nil {
			continue
		}
		if err := gbbqEnc.Encode(GbbqRow{
			Date:         event.Time.Format(time.DateOnly),
			Symbol:       etf.Symbol,
			TDXSymbol:    code,
			CategoryCode: event.Category,
			C1:           event.C1,
			C2:           event.C2,
			C3:           event.C3,
			C4:           event.C4,
		}); err != nil {
			return fmt.Errorf("write gbbq: %w", err)
		}
	}

	xrxds := protocol.XRXDs{}
	for _, event := range gbbqResp.List {
		if event != nil && event.IsXRXD() {
			xrxds = append(xrxds, event.XRXD())
		}
	}

	sort.Slice(resp.List, func(i, j int) bool { return resp.List[i].Time.Before(resp.List[j].Time) })
	factors := xrxds.Pre(resp.List).Factors()
	qfq := protocol.ApplyQFQ(resp.List, factors)

	if len(qfq) != len(resp.List) {
		return fmt.Errorf("raw/qfq length mismatch: raw=%d qfq=%d", len(resp.List), len(qfq))
	}

	for i, raw := range resp.List {
		if raw == nil || raw.Time.Before(startDate) {
			continue
		}
		if err := bfqEnc.Encode(dailyRow(etf, code, raw)); err != nil {
			return fmt.Errorf("write bfq: %w", err)
		}
		if err := qfqEnc.Encode(dailyRow(etf, code, qfq[i])); err != nil {
			return fmt.Errorf("write qfq: %w", err)
		}
	}

	return nil
}

func dailyRow(etf ETF, code string, k *protocol.Kline) DailyRow {
	return DailyRow{
		Date:      k.Time.Format(time.DateOnly),
		Symbol:    etf.Symbol,
		TDXSymbol: code,
		Name:      etf.Name,
		Category:  etf.Category,
		Bucket:    etf.Bucket,
		Open:      k.Open.Float64(),
		High:      k.High.Float64(),
		Low:       k.Low.Float64(),
		Close:     k.Close.Float64(),
		Volume:    k.Volume,
		Amount:    k.Amount.Float64(),
	}
}

func tdxSymbol(symbol string) string {
	if strings.HasPrefix(symbol, "5") || strings.HasPrefix(symbol, "6") {
		return "sh" + symbol
	}
	if strings.HasPrefix(symbol, "0") || strings.HasPrefix(symbol, "1") || strings.HasPrefix(symbol, "3") {
		return "sz" + symbol
	}
	fatalf("cannot infer exchange prefix for %s", symbol)
	return ""
}

func mustCreate(path string) *os.File {
	f, err := os.Create(path)
	if err != nil {
		fatalf("create %s: %v", path, err)
	}
	return f
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
