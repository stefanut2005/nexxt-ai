using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Net.Http.Json;
using FraudDashboard.Models;

namespace FraudDashboard.Pages;

public class IndexModel : PageModel
{
    public List<Transaction> Transactions { get; set; } = new();

    public async Task OnGet()
    {
        ViewData["Title"] = "Overview";
        ViewData["Subtitle"] = "Live fraud risk across recent transactions";

        using var client = new HttpClient();
        var resp = await client.GetAsync("http://localhost:5000/proxy/transactions");
        if (resp.IsSuccessStatusCode)
        {
            var data = await resp.Content.ReadFromJsonAsync<List<Transaction>>();
            if (data != null)
                Transactions = data;
        }
    }
}
