
import Chart from 'chart.js/auto'

import { rep_revenue_projections } from './api'

(async function() {

  const data =  await rep_revenue_projections();

    new Chart(
      document.getElementById('acquisitions'),
      {
        type: 'line',
        data: {
          labels: data.map(row => row.year),
          datasets: [
            {
              label:'x' ,
              data: 'x'
            }
          ]
        }
      }
    );
})();