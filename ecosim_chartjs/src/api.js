import { CubejsApi } from '@cubejs-client/core';
const apiUrl = 'xx';
const cubeToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjEwMDAwMDAwMDAsImV4cCI6NTAwMDAwMDAwMH0.OHZOpOBVKr-sCwn8sbZ5UFsqI3uCs6e4omT7P6WVMFw';
const cubeApi = new CubejsApi(cubeToken, { apiUrl });

export async function government() {
    const governmentQuery = {
        dimensions: [
        'Artworks.year',
    ],
    measures: [
        'Artworks.population',
        'Artworks.GDP'
    ],
    };
    const resultSet = await cubeApi.load(acquisitionsByYearQuery);

    return resultSet.tablePivot().map(row => ({
        year: parseInt(row['Artworks.year']),
        gdp: parseInt(row['Artworks.count']),
        population: parseInt(row['Artworks.population']),
    }));
}


export async function corperation() {
    const corperationQuery = {
        dimensions: [
        'Artworks.year',
    ],
    measures: [

    ],
    };
    const resultSet = await cubeApi.load(acquisitionsByYearQuery);

    return resultSet.tablePivot().map(row => ({
        year: parseInt(row['Artworks.year']),

    }));
}


export async function workers() {
    const workersQuery = {
        dimensions: [
        'Artworks.year',
    ],
    measures: [
        'Artworks.compensation',
        'Artworks.enjoyment'
    ],
    };
    const resultSet = await cubeApi.load(acquisitionsByYearQuery);

    return resultSet.tablePivot().map(row => ({
        year: parseInt(row['Artworks.year']),
        compensation: parseInt(row['Artworks.compensation']),
        enjoyment: parseInt(row['Artworks.enjoyment']),
    }));
}