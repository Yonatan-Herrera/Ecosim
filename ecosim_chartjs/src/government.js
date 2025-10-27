
import Chart from 'chart.js/auto'

import { government } from './api'

(async function() {

  const data =  await government();

    new Chart(
      document.getElementById('acquisitions'),
      {
        type: 'line',
        data: {
          labels: data.map(row => row.year),
          datasets: [
            {
              label: 'GDPP per year',
              data: data.map(row => row.gdp/row.population)              
            }
          ]
        }
      }
    );
})();