using System.Net.Http.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorPages()
	.AddRazorRuntimeCompilation();

builder.Services.AddHttpClient("PythonApi", client =>
{
	var pythonApiBase = builder.Configuration["PythonApi:BaseUrl"]
						?? Environment.GetEnvironmentVariable("PYTHON_API_BASE")
						?? "http://localhost:5001";
	client.BaseAddress = new Uri(pythonApiBase);
});

builder.Services.AddCors(options =>
{
	options.AddPolicy("AllowAll", p =>
		p.AllowAnyOrigin()
		 .AllowAnyHeader()
		 .AllowAnyMethod());
});

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
	app.UseExceptionHandler("/Error");
	app.UseHsts();
	app.UseHttpsRedirection();
}
else
{
	app.Logger.LogInformation("Running in Development: HTTPS redirect is disabled.");
}

app.UseStaticFiles();
app.UseRouting();
app.UseCors("AllowAll");

app.MapGet("/proxy/transactions", async (IHttpClientFactory httpFactory) =>
{
	var client = httpFactory.CreateClient("PythonApi");
	var result = await client.GetFromJsonAsync<List<object>>("/transactions");
	return Results.Json(result);
});

app.MapPost("/proxy/chat", async (HttpRequest request, IHttpClientFactory httpFactory) =>
{
	var client = httpFactory.CreateClient("PythonApi");
	using var bodyReader = new StreamReader(request.Body);
	var rawBody = await bodyReader.ReadToEndAsync();

	using var forwardMsg = new HttpRequestMessage(HttpMethod.Post, "/chat")
	{
		Content = new StringContent(rawBody, System.Text.Encoding.UTF8, "application/json")
	};

	var resp = await client.SendAsync(forwardMsg);
	var payload = await resp.Content.ReadAsStringAsync();
	return Results.Content(payload, "application/json");
});

app.MapRazorPages();
app.Run();
