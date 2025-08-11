const stockSymbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"];

// 1) Fetch stock data (example using Yahoo Finance API endpoints)
async function fetchStockData() {
  const symbolsParam = stockSymbols.join(",");
  const url = `https://query1.finance.yahoo.com/v7/finance/quote?symbols=${symbolsParam}`;

  const resp = await fetch(url);
  const data = await resp.json();
  // Return in a format that Chart.js can digest
  return data.quoteResponse.result.map(item => ({
    symbol: item.symbol,
    price: item.regularMarketPrice
  }));
}

// 2) Create a simple line chart for current prices (mocked with same price)
async function renderStockChart() {
  const stockData = await fetchStockData();
  const ctx = document.getElementById("stocksChart").getContext("2d");

  const labels = Array.from({ length: 24 }, (_, i) => `${i}h`);
  const datasets = stockData.map((stock, idx) => ({
    label: stock.symbol,
    data: new Array(24).fill(stock.price), // Replace with real historical data
    borderColor: `hsl(${(idx * 60) % 360}, 70%, 50%)`,
    fill: false
  }));

  new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#fff" } } },
      scales: {
        x: { ticks: { color: "#fff" }, grid: { color: "rgba(255,255,255,0.1)" } },
        y: { ticks: { color: "#fff" }, grid: { color: "rgba(255,255,255,0.1)" } }
      }
    }
  });
}

// 3) Fetch latest market news from Yahoo Finance RSS feed
async function fetchArticles() {
  const rssUrl = "https://finance.yahoo.com/news/rssindex"; // Example RSS feed
  const resp = await fetch(rssUrl);
  const text = await resp.text();

  // Minimal RSS parsing using DOMParser (client-side, may be CORS blocked on some browsers)
  const parser = new DOMParser();
  const doc = parser.parseFromString(text, "application/xml");
  const items = Array.from(doc.querySelectorAll("item")).slice(0, 4);

  return items.map(item => ({
    title: item.querySelector("title")?.textContent ?? "",
    link: item.querySelector("link")?.textContent ?? ""
  }));
}

async function renderArticles() {
  const articles = await fetchArticles();
  const grid = document.getElementById("articlesGrid");

  articles.forEach(article => {
    const card = document.createElement("div");
    card.className = "article-card";
    card.innerHTML = `
      <h3><a href="${article.link}" target="_blank">${article.title}</a></h3>
    `;
    grid.appendChild(card);
  });
}

renderStockChart();
renderArticles();
