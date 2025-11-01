namespace FraudDashboard.Models;

public class Transaction
{
    public string Id { get; set; } = string.Empty;
    public string CardLast4 { get; set; } = string.Empty;
    public string Merchant { get; set; } = string.Empty;
    public decimal Amount { get; set; }
    public string Currency { get; set; } = "EUR";
    public DateTime Timestamp { get; set; }
    public double Risk { get; set; } // 0-100
    public string RiskLabel { get; set; } = string.Empty;
    public string Location { get; set; } = string.Empty;
    public string Notes { get; set; } = string.Empty;
    public Dictionary<string, object>? Features { get; set; }
}
