(async function(){
    const highRiskEl = document.getElementById('highRiskCount');
    const avgRiskEl  = document.getElementById('avgRisk');

    try {
        const resp = await fetch('/proxy/transactions');
        if(!resp.ok) return;
        const data = await resp.json();
        if(!Array.isArray(data)) return;

        const risks = data.map(t => t.risk ?? t.Risk ?? 0);
        const highRisk = risks.filter(r => r >= 80).length;
        const avgRisk = risks.length ? (risks.reduce((a,b)=>a+b,0)/risks.length) : 0;

        if (highRiskEl) highRiskEl.textContent = highRisk.toString();
        if (avgRiskEl)  avgRiskEl.textContent  = avgRisk.toFixed(1) + '%';
    } catch (err){
        console.error('dashboard stats error', err);
    }
})();

// THEME TOGGLE
(function(){
    const btn = document.getElementById('themeToggle');
    if(!btn) return;

    function currentTheme(){
        return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    }

    function apply(theme){
        if(theme === 'dark'){
            document.documentElement.setAttribute('data-theme','dark');
            btn.textContent = '☀️';
        } else {
            document.documentElement.removeAttribute('data-theme');
            btn.textContent = '🌙';
        }
    }

    apply(currentTheme());

    btn.addEventListener('click', ()=>{
        const next = currentTheme() === 'dark' ? 'light' : 'dark';
        apply(next);
        localStorage.setItem('fd-theme', next);
    });
})();
