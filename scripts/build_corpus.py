"""
scripts/build_corpus.py
Build corpus_node1.json / corpus_node2.json from hand-curated source paragraphs.

global_id: content-addressed ID = "gid_" + sha256(f"{source}|{author}|{date}")[:16]
(canonical metadata hash, per the corpus-versioning decision in docs/GIN_corpus_sources.md).
chunk granularity: paragraph-level (matches Phase 1 synthetic corpus format).

Run: python scripts/build_corpus.py [--node 1|2|all]
Writes corpus_node1.json and/or corpus_node2.json to the repo root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def global_id(source: str, author: str, date: str) -> str:
    canonical = f"{source}|{author}|{date}"
    return "gid_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_doc(idx, node_id, source, url, author, date, dtype, category, domain, paras):
    # Namespace doc ids by node so both manifests can be ingested into one DB
    # without colliding on doc_id / chunk_id (warm chunk id is "<doc_id>:<index>").
    prefix = "n1_" if node_id.startswith("node_1") else "n2_"
    doc_id = f"{prefix}doc_{idx:03d}"
    chunks = [
        {"chunk_id": f"{doc_id}_c{pos:03d}", "position": pos, "text": text}
        for pos, text in enumerate(paras)
    ]
    return {
        "doc_id": doc_id,
        "global_id": global_id(source, author, date),
        "source": source,
        "url": url,
        "node": node_id,
        "metadata": {
            "domain": domain,
            "date": date,
            "type": dtype,
            "author": author,
            "category": category,
        },
        "chunks": chunks,
    }


# ---------------------------------------------------------------- NODE 1
N1 = "node_1_institutional"
DOM1 = "environmental_measurement"
node1_docs = [
    build_doc(1, N1,
        "IPCC AR6 Synthesis Report, Summary for Policymakers",
        "https://www.ipcc.ch/report/ar6/syr/summary-for-policymakers/",
        "IPCC", "2023-03", "institutional_report", "climate_synthesis", DOM1,
        [
            "Human activities, principally through emissions of greenhouse gases, have unequivocally caused global warming, with global surface temperature reaching 1.1 degrees C above 1850-1900 in 2011-2020.",
            "Global surface temperature has increased faster since 1970 than in any other 50-year period over at least the last 2000 years.",
            "Atmospheric CO2 concentrations in 2019 were higher than at any time in at least 2 million years, while methane and nitrous oxide concentrations were higher than at any time in at least 800,000 years.",
            "Widespread and rapid changes in the atmosphere, ocean, cryosphere and biosphere have occurred, with human-caused climate change already affecting many weather and climate extremes in every region.",
            "Approximately 3.3 to 3.6 billion people live in contexts that are highly vulnerable to climate change, with the largest adverse impacts observed in Africa, Asia, Central and South America, and among Indigenous Peoples and low-income households.",
            "Climate change has caused substantial damages and increasingly irreversible losses in terrestrial, freshwater, cryospheric, and coastal ecosystems, with hundreds of local species extinctions driven by heat extremes.",
            "Adaptation planning and implementation has progressed across all sectors and regions, but adaptation gaps exist and will continue to grow at current implementation rates.",
            "Global greenhouse gas emissions in 2019 reached roughly 59 GtCO2-eq, about 54% higher than in 1990, with the largest share from CO2 emissions from fossil fuels and industrial processes.",
        ]),
    build_doc(2, N1,
        "NOAA Annual 2023 Global Climate Report",
        "https://www.ncei.noaa.gov/news/global-climate-202312",
        "NOAA NCEI", "2023-12", "institutional_report", "temperature_data", DOM1,
        [
            "NOAA ranks 2023 as the warmest year in its global temperature record, which dates back to 1850.",
            "In 2023, global surface temperature was about 2.12 degrees F (1.18 degrees C) above the 20th-century average, beating the next warmest year (2016) by roughly 0.27 degrees F.",
            "The 10 warmest years since 1850 have all occurred in the past decade.",
            "Upper ocean heat content, the amount of heat stored in the top 2000 meters of the ocean, was record high in 2023.",
            "Arctic sea ice extent averaged about 4.05 million square miles in 2023, ranking among the 10 lowest years on record.",
            "Antarctic sea ice extent averaged about 3.79 million square miles in 2023, the lowest on record.",
            "Seventy-eight named storms occurred globally in 2023, below the 1991-2020 average of 87.5.",
            "Global surface temperature in December 2023 was about 2.57 degrees F above the 20th-century average, the warmest December on record.",
        ]),
    build_doc(3, N1,
        "NASA GISS Surface Temperature Analysis (GISTEMP v4)",
        "https://data.giss.nasa.gov/gistemp/",
        "NASA GISS", "2023-06", "explanatory_article", "temperature_data", DOM1,
        [
            "The GISS Surface Temperature Analysis version 4 (GISTEMP v4) is an estimate of global surface temperature change.",
            "Graphs and tables are updated about the 10th of every month using current data files from NOAA GHCN v4 (meteorological stations) and ERSST v5 (ocean areas).",
            "The basic GISS temperature analysis scheme was defined in the late 1970s by James Hansen when a method of estimating global temperature change was needed for comparison with one-dimensional global climate models.",
            "The analysis method was fully documented in Hansen and Lebedeff (1987), with several papers describing updates to the analysis following over the decades.",
            "Monthly surface temperature anomaly data are available in several forms, including plain-text tables of Land-Ocean Temperature Index deviations, netCDF gridded data, and compressed Zarr data directories.",
            "Programs used in the GISTEMP analysis assume a Unix-like operating system and require familiarity with Python.",
            "Datasets and source code for uncertainty analyses are available for the 2024 uncertainty ensemble and the 2019 uncertainty quantification analysis by Lenssen et al.",
            "The GISTEMP analysis was initiated by Dr. James E. Hansen and is currently led by Dr. Gavin Schmidt.",
        ]),
    build_doc(4, N1,
        "NOAA Ocean Acidification Explainer",
        "https://oceanservice.noaa.gov/facts/acidification.html",
        "NOAA", "2023-06", "institutional_summary", "ocean_acidification", DOM1,
        [
            "Ocean acidification refers to a reduction in the pH of the ocean over an extended period of time, caused primarily by uptake of carbon dioxide (CO2) from the atmosphere.",
            "For more than 200 years since the industrial revolution, atmospheric CO2 concentration has increased due to fossil fuel burning and land use changes, with the ocean absorbing roughly 30 percent of released CO2.",
            "When CO2 dissolves in seawater, chemical reactions increase hydrogen ion concentration, making the water more acidic and reducing the abundance of carbonate ions.",
            "Carbonate ions serve as building blocks for shells and coral skeletons; their depletion makes it harder for calcifying organisms like oysters, clams, sea urchins, and corals to build structures.",
            "Ocean acidification also affects non-calcifying organisms, such as fish whose predator-detection abilities are diminished in more acidic conditions, potentially destabilizing food webs.",
            "The phenomenon affects all of the world's oceans, including coastal estuaries and waterways where many economies depend on fish and shellfish resources.",
            "Billions of people rely on ocean protein sources, making ocean acidification a significant threat to global food security and economic stability.",
        ]),
    build_doc(5, N1,
        "UNEP Emissions Gap Report 2023: Broken Record",
        "https://www.unep.org/resources/emissions-gap-report-2023",
        "UNEP", "2023-11", "institutional_report", "emissions_assessment", DOM1,
        [
            "In 2023, 86 days were recorded with temperatures more than 1.5 degrees C above pre-industrial levels; September was the hottest month on record, with global average temperatures about 1.8 degrees C above pre-industrial levels.",
            "Current pledges under the Paris Agreement put the world on track for a 2.5-2.9 degree C temperature rise above pre-industrial levels this century.",
            "Global low-carbon transformations are needed to deliver cuts to predicted 2030 greenhouse gas emissions of roughly 28 percent for a 2 degree C pathway and 42 percent for a 1.5 degree C pathway.",
            "Projected 2030 emissions growth relative to policies in place has fallen from 16 percent at the time of the Paris Agreement's adoption to about 3 percent today.",
            "Released in November 2023, the Emissions Gap Report finds that current pledges point to the urgent need for increased climate action to close the emissions gap.",
        ]),
    build_doc(6, N1,
        "WMO State of the Global Climate 2023",
        "https://wmo.int/publication-series/state-of-global-climate-2023",
        "WMO", "2024-03", "institutional_report", "climate_assessment", DOM1,
        [
            "The State of the Global Climate 2023 report shows that records were broken for greenhouse gas levels, surface temperatures, ocean heat and acidification, sea level rise, Antarctic sea ice cover and glacier retreat.",
            "Heatwaves, floods, droughts, wildfires, and rapidly intensifying tropical cyclones caused widespread disruption and inflicted many billions of dollars in economic losses in 2023.",
            "The WMO report confirmed 2023 as the warmest year on record, with the global average near-surface temperature about 1.45 degrees C above the pre-industrial baseline.",
            "Records were broken for ocean heat, sea level rise, Antarctic sea ice loss, and glacier retreat.",
        ]),
    build_doc(7, N1,
        "EPA Climate Change Indicators: U.S. and Global Temperature",
        "https://www.epa.gov/climate-indicators/climate-change-indicators-us-and-global-temperature",
        "EPA", "2024-06", "institutional_indicator", "climate_metrics", DOM1,
        [
            "Worldwide, 2023 was the warmest year on record and 2014-2023 was the warmest decade on record since thermometer-based observations began; global average surface temperature has risen about 0.15 degrees F per decade since 1901.",
            "Since 1901, the average surface temperature across the contiguous 48 states has risen about 0.14 degrees F per decade, with faster warming since the late 1970s.",
            "Eight of the top 10 warmest years on record for the contiguous 48 states have occurred since 1998.",
            "In the U.S., unusually hot summer days have become more common over recent decades, and unusually hot summer nights have increased at an even faster rate.",
            "Concentrations of heat-trapping greenhouse gases are increasing in the Earth's atmosphere, and average surface temperatures are expected to continue rising in response.",
        ]),
    build_doc(8, N1,
        "NIFC National Interagency Coordination Center 2023 Wildland Fire Summary",
        "https://www.nifc.gov/fire-information/statistics/wildfires",
        "National Interagency Fire Center", "2024-01", "institutional_report", "wildfire", DOM1,
        [
            "In 2023, 56,580 wildfires burned 2,693,910 acres across the United States, with acreage burned below both the five- and ten-year averages.",
            "There were 891 large wildfires and complexes reported in 2023, representing less than 2 percent of total wildfires reported nationally.",
            "About one-quarter of the nation's wildfires in 2023 occurred on federally protected lands.",
            "A total of 4,318 structures were destroyed by wildfires in 2023, including 3,060 residences, 1,228 minor structures, and 51 commercial or mixed-residential structures.",
            "National wildland fire preparedness levels and interagency suppression resources are coordinated through the National Interagency Coordination Center based on acreage, active incident counts, and resource demand.",
        ]),
    build_doc(9, N1,
        "California DWR 2023 Snow Survey and Reservoir Conditions",
        "https://water.ca.gov/News/News-Releases/2023/April-23/Snow-Survey-April-2023",
        "California Department of Water Resources", "2023-04", "institutional_report", "water_scarcity", DOM1,
        [
            "As of April 3, 2023, California's statewide snowpack held a snow water equivalent of 61.1 inches, or 237 percent of the April 1 average, one of the largest snowpacks on record.",
            "By May 1, 2023, the statewide snow water equivalent measured 49.2 inches, or 254 percent of average for that date.",
            "The Southern Sierra snowpack was 300 percent of its April 1 average, the Central Sierra was 237 percent, and the Northern Sierra, where the state's largest surface reservoirs are located, was 192 percent.",
            "The State Water Project's two largest reservoirs, Oroville and San Luis, gained a combined 1.62 million acre-feet of water in storage.",
            "Only 1952, 1969, and 1983 previously recorded statewide April 1 results above 200 percent of average, framing 2023 as a historic year for measured water supply.",
        ]),
]

# ---------------------------------------------------------------- NODE 2
N2 = "node_2_grassroots"
DOM2 = "environmental_impact"
node2_docs = [
    build_doc(1, N2,
        "Indigenous Environmental Network: Frontline Communities Demand Real Climate Solutions",
        "https://www.ienearth.org/",
        "Indigenous Environmental Network", "2023-12", "advocacy_report", "indigenous_leadership", DOM2,
        [
            "People living on the frontlines of climate chaos and the fossil fuel industry are disproportionately Indigenous Peoples, Black and Brown communities, low-wage workers, and smallholder farmers, often living in poverty.",
            "Frontline delegations call on world leaders to pass binding agreements, including an immediate phase-out of dirty energy, and to commit to meaningful climate reparations for communities bearing the brunt of the climate crisis.",
            "Advocates reject market-based schemes and techno-fixes they say are designed to prolong the fossil fuel industry's lifespan and put communities at risk.",
            "Critics argue current climate finance mechanisms do not give frontline communities direct, no-strings-attached access to funds, and instead benefit the Global North at the expense of impacted communities.",
            "Indigenous-led resistance efforts are estimated to have stopped or delayed greenhouse gas pollution equivalent to roughly one-quarter of annual U.S. and Canadian emissions.",
        ]),
    build_doc(2, N2,
        "NDN Collective: Climate Justice and LANDBACK",
        "https://ndncollective.org/climate-justice/",
        "NDN Collective", "2023-06", "advocacy_report", "indigenous_leadership", DOM2,
        [
            "NDN Collective is an Indigenous-led organization focused on building Indigenous power through organizing, philanthropy, and narrative change to advance solutions on Indigenous terms.",
            "Its climate justice work supports Tribes and Indigenous Peoples in pursuing a justice-based transition, including engagement with federal clean-energy and infrastructure funding opportunities.",
            "Its campaigns center Indigenous and frontline communities defending land, water, and air from contamination and resisting exploitation by the fossil fuel industry.",
            "LANDBACK is described as an organizing and narrative framework for Indigenous peoples to reclaim stewardship of land and move toward sovereignty.",
            "The organization supports economic practices intended to maintain and expand Indigenous land base, sovereignty, and rights, shifting decision-making power to Indigenous peoples.",
        ]),
    build_doc(3, N2,
        "ProPublica: Climate Change Will Force a New American Migration",
        "https://www.propublica.org/article/climate-change-will-force-a-new-american-migration",
        "ProPublica", "2020-09", "investigative_journalism", "climate_migration", DOM2,
        [
            "Extreme heat events, such as California's August 2020 heat wave that strained the electrical grid and coincided with record temperatures near Death Valley, illustrate the lived disruption of a changing climate.",
            "Analysis suggests roughly 162 million Americans, nearly half the population, are likely to experience declining environmental quality from more heat and less water; under high-emissions scenarios, millions could live in conditions outside the historical human comfort niche by 2070.",
            "One study projects that a substantial share of Americans in the South could migrate toward California, the Mountain West, or the Northwest over the coming decades due to climate pressures, a shift that could widen economic inequality.",
            "Regional warming is projected to reshape climate norms across the country, with severe water shortages becoming common west of the Missouri River by 2040 under federal projections.",
            "Researchers estimate millions of Americans could be forced to relocate from vulnerable coastlines, with additional displacement likely from wildfire and other climate risks.",
            "Such a shift, if realized, would represent one of the largest internal migrations in American history, comparable in scale to the Great Migration of the 20th century.",
            "Models suggest migration would concentrate in Northeastern and Northwestern cities, with population growth making historically cold regions more temperate.",
            "Reporting emphasizes that most people do not choose to leave home voluntarily and tend to relocate only when other options are exhausted.",
        ]),
    build_doc(4, N2,
        "Inside Climate News: On a 'Toxic Tour' of Curtis Bay, South Baltimore",
        "https://insideclimatenews.org/news/06082023/baltimore-harm-cityenvironmental-justice-neighborhoods/",
        "Inside Climate News", "2023-08", "investigative_journalism", "frontline_impact", DOM2,
        [
            "South Baltimore's Curtis Bay neighborhood has been described as a 'sacrifice zone' due to nearby coal export terminals, with coal dust linked to elevated asthma and respiratory illness rates among residents.",
            "Communities such as Cameron Parish, Louisiana, and Port Arthur, Texas, are cited as environmental justice hotspots where Black and Brown residents face disproportionate pollution exposure.",
            "Reporting cites industrial cancer-risk estimates in Port Arthur far above levels the EPA considers acceptable, with the area ranking in the highest percentile nationally for toxic air releases.",
            "Frontline communities across the Gulf South and into Canada are reported to face significant impacts from fossil fuel infrastructure projects backed by major financial firms.",
        ]),
    build_doc(5, N2,
        "Earthjustice: How We Can Preserve Breathable Air in a World on Fire",
        "https://earthjustice.org/article/how-we-can-preserve-breathable-air-in-a-world-on-fire",
        "Earthjustice", "2023-06", "advocacy_explainer", "environmental_justice", DOM2,
        [
            "Research cited by Earthjustice finds that Black and Brown residents in the U.S. breathe more soot pollution on average than white residents, largely because pollution sources are sited disproportionately near their communities.",
            "Elderly, immunocompromised, and low-income populations face heightened risk from wildfire smoke exposure.",
            "Climate change, including more extreme heat and larger wildfires, is identified as a driver of rising ozone and particulate pollution.",
            "Advocacy groups point to grant programs intended to help communities engage in environmental review processes and reduce health disparities in frontline, pollution-burdened areas.",
        ]),
    build_doc(6, N2,
        "Union of Concerned Scientists: Environmental Justice for All Act Historic for Frontline Communities",
        "https://www.ucsusa.org/about/news/environmental-justice-all-act-historic-frontline-communities",
        "Union of Concerned Scientists", "2022-07", "advocacy_report", "environmental_justice", DOM2,
        [
            "UCS environmental justice work centers on the cumulative health burden faced by communities exposed to multiple, overlapping pollution sources.",
            "Communities near new pipeline and energy infrastructure projects, disproportionately Black, Brown, and Indigenous, already face compounding pollution burdens.",
            "Fossil fuel power plants are described as continuing to harm nearby communities, which also face climate-driven risks such as sea level rise, flooding, and worsening wildfire seasons.",
            "Advocates have called for directing a substantial share, roughly 40 percent, of federal clean energy and climate resilience investment toward communities of color and low-income communities.",
        ]),
    build_doc(7, N2,
        "WE ACT for Environmental Justice: Origins in Harlem Air Quality Advocacy",
        "https://weact.org/",
        "WE ACT for Environmental Justice", "2023-06", "advocacy_brief", "environmental_justice", DOM2,
        [
            "WE ACT for Environmental Justice is a community-based organization founded in 1988 as West Harlem Environmental Action, working to ensure people of color and low-income residents participate meaningfully in environmental policymaking.",
            "The organization traces its founding to a 1988 protest by Harlem residents against air-quality threats posed by the North River Sewage Treatment Plant.",
            "Organizers involved in that protest, arrested for civil disobedience, ultimately succeeded in reducing the community's exposure to related toxins and hazards.",
            "WE ACT continues to work with members and community stakeholders on environmental justice advocacy campaigns at the local, state, and federal levels.",
        ]),
    build_doc(8, N2,
        "Pacific Institute: Drought and Equity in California",
        "https://pacinst.org/publication/drought-and-equity-in-california/",
        "Pacific Institute", "2022-06", "research_report", "water_scarcity", DOM2,
        [
            "Pacific Institute research finds California could meaningfully reduce urban water use through greater investment in water efficiency measures.",
            "Strategies such as water efficiency, reuse, and stormwater capture are described as already succeeding in many California communities, reducing demand while boosting local supply.",
            "Disadvantaged and cumulatively burdened communities are found to be disproportionately affected by water shortages, reflecting underlying inequities in water resource management.",
            "Community-based participatory research on drought impacts in the San Francisco Bay Area highlights affordability and infrastructure condition as persistent concerns for low-income residents.",
            "Per-capita water use is generally found to be higher in wealthier communities.",
        ]),
    build_doc(9, N2,
        "Grist: How Environmental Justice Leaders Are Pushing Forward",
        "https://grist.org/justice/heres-how-environmental-justice-leaders-are-pushing-forward-in-the-trump-era/",
        "Grist", "2021-06", "investigative_journalism", "environmental_justice", DOM2,
        [
            "Grist's environmental justice reporting profiles frontline leaders and their advocacy work at international climate negotiations.",
            "Coverage examines cases where environmental improvements and community advocacy victories have inadvertently been used by developers to justify displacing existing residents.",
            "Reporting stresses the need to evaluate wildfire impacts on disadvantaged communities through an environmental justice lens.",
        ]),
    build_doc(10, N2,
        "Chesapeake Bay Foundation: State of the Bay Report",
        "https://www.cbf.org/about-the-bay/state-of-the-bay-report/",
        "Chesapeake Bay Foundation", "2022-12", "research_report", "place_based_impact", DOM2,
        [
            "From 1998 to 2022, CBF's State of the Bay report tracked the health of the Chesapeake Bay and progress toward restoration goals.",
            "The organization is currently revising its report format and methodology for future assessments.",
            "The University of Maryland Center for Environmental Science separately publishes annual Bay watershed report cards.",
            "The 2022 CBF assessment assigned the Bay an overall score of 32, a D+ grade.",
            "The 2020 report similarly showed a score of 32, also a D+.",
            "The 2018 report gave the Bay a score of 33, also rated D+.",
            "The 2016 evaluation showed slight improvement, with a score of 34, a C- grade.",
            "These reports track bay health through a consistent set of environmental indicators measured over multiple decades.",
        ]),
]

ALL_NODES = [
    (N1, node1_docs, "corpus_node1.json"),
    (N2, node2_docs, "corpus_node2.json"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node",
        choices=["1", "2", "all"],
        default="all",
        help="Which node corpus to generate (default: all)",
    )
    args = parser.parse_args()

    if args.node == "all":
        targets = ALL_NODES
    else:
        targets = [ALL_NODES[int(args.node) - 1]]

    for node_id, docs, fname in targets:
        payload = {"node_id": node_id, "documents": docs}
        path = ROOT / fname
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        n_chunks = sum(len(d["chunks"]) for d in docs)
        print(f"{fname}: {len(docs)} docs, {n_chunks} chunks")
        for d in docs:
            print(f"  {d['doc_id']}  {d['global_id']}  {len(d['chunks'])}c  {d['source'][:50]}")


if __name__ == "__main__":
    main()
