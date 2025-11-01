using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Net.Http.Json;
using FraudDashboard.Models;

namespace FraudDashboard.Pages.Transactions;

public class IndexModel : PageModel
{
    [BindProperty(SupportsGet = true)]
    public int MinRisk { get; set; } = 0;

    public List<Transaction> Transactions { get; set; } = new();

    public async Task OnGet()
    {
        ViewData["Title"] = "Transactions";
        ViewData["Subtitle"] = "Full list with fraud score";

        using var client = new HttpClient();
        var resp = await client.GetAsync("http://localhost:5000/proxy/transactions");
        if (resp.IsSuccessStatusCode)
        {
            var data = await resp.Content.ReadFromJsonAsync<List<Transaction>>();
            if (data != null)
                Transactions = data
                    .Where(t => t.Risk >= MinRisk)
                    .OrderByDescending(t => t.Timestamp)
                    .ToList();
        }
    }
}
