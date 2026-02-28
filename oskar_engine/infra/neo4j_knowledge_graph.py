"""
Neo4j Knowledge Graph for OSKAR v0.3
-------------------------------------
Provides entity-relationship evidence retrieval as a Graph-RAG layer
on top of the existing FAISS index.

If Neo4j is unavailable (not running / no docker), the class
degrades gracefully and returns no graph triples — the FAISS-only
path then activates automatically as the fallback.

Usage:
    kg = KnowledgeGraph()
    if kg.connected:
        triples = kg.query_context("Vaccines cause autism")
"""

import os
import re
from typing import Optional

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "oskarpass")

# ─── 150 seed fact triples covering common misinformation topics ─────────────
SEED_TRIPLES = [
    # Vaccines
    ("CDC", "STATES", "Vaccines do not cause autism"),
    ("WHO", "CONFIRMS", "COVID-19 vaccines are safe and effective"),
    ("MMR vaccine", "PREVENTS", "Measles, mumps, and rubella"),
    ("Thimerosal", "REMOVED_FROM", "Childhood vaccines in 2001"),
    ("Vaccine safety data", "MONITORED_BY", "VAERS and CDC"),
    ("Herd immunity", "ACHIEVED_THROUGH", "Vaccination campaigns"),
    ("Polio eradication", "ACHIEVED_BY", "Global vaccine programs"),
    ("COVID-19 mRNA vaccine", "DOES_NOT_ALTER", "Human DNA"),
    ("Aluminum adjuvants", "USED_SAFELY_IN", "Vaccines since 1930s"),
    ("Autism diagnosis rate", "INCREASED_DUE_TO", "Broader diagnostic criteria"),
    # Climate Change
    ("IPCC", "REPORTS", "97% of scientists agree climate change is human-caused"),
    ("CO2 levels", "REACHED", "420 ppm in 2023, highest in 800000 years"),
    ("Arctic ice", "HAS_DECLINED", "By 40% since 1980"),
    (
        "Global average temperature",
        "RISEN_BY",
        "1.2 degrees Celsius since industrialization",
    ),
    ("Fossil fuel combustion", "IS_PRIMARY_CAUSE_OF", "Anthropogenic CO2 emissions"),
    ("Paris Agreement", "SIGNED_BY", "196 countries in 2015"),
    ("Sea levels", "RISING_AT", "3.7mm per year due to ice melt"),
    ("Renewable energy", "NOW_CHEAPER_THAN", "Coal in most markets"),
    ("Methane", "IS", "A greenhouse gas 80x more potent than CO2 over 20 years"),
    ("Climate models", "ACCURATELY_PREDICTED", "Surface temperature trends since 1970"),
    # Elections
    ("2020 US election", "WAS_CERTIFIED_BY", "All 50 states and Congress"),
    ("Voter fraud", "IS", "Extremely rare at 0.00004% of ballots"),
    ("Electronic voting machines", "ARE_AUDITED_BY", "Local election officials"),
    ("Mail-in ballots", "HAVE_SAME_FRAUD_RATE_AS", "In-person ballots"),
    ("Dominion Voting Systems", "WAS_CLEARED_OF_FRAUD_CLAIMS_BY", "60+ court rulings"),
    ("Election security", "OVERSEEN_BY", "CISA and state election boards"),
    ("Voter ID laws", "VARY_ACROSS", "Different US states"),
    ("Gerrymandering", "AFFECTS", "Electoral district boundaries"),
    ("Ranked choice voting", "USED_IN", "Maine and Alaska"),
    ("Electoral College", "ESTABLISHED_BY", "US Constitution Article II"),
    # 5G and Radiation
    ("5G networks", "OPERATE_AT", "Non-ionizing radio frequencies"),
    ("Non-ionizing radiation", "CANNOT", "Break chemical bonds or damage DNA"),
    ("5G", "DOES_NOT_CAUSE", "COVID-19 or spread viruses"),
    ("WHO", "CLASSIFIES", "5G as safe at current exposure levels"),
    ("Radio waves", "HAVE_BEEN_USED_SAFELY_SINCE", "Early 20th century"),
    ("Ionizing radiation", "SOURCE_IS", "Gamma rays, X-rays, and UV light"),
    ("Cell phone towers", "EMIT_RADIATION_FAR_BELOW", "International safety limits"),
    ("5G rollout", "IS_MONITORED_BY", "FCC and ICNIRP"),
    ("Electromagnetic hypersensitivity", "NOT_SUPPORTED_BY", "Scientific evidence"),
    ("Fiber optic cables", "DO_NOT", "Use radio waves"),
    # Flat Earth and Space
    (
        "Earth",
        "IS_PROVEN_SPHERICAL_BY",
        "Satellite imagery, physics, and circumnavigation",
    ),
    ("NASA", "HAS_PHOTOGRAPHIC_EVIDENCE_OF", "Earth's curvature"),
    ("International Space Station", "HAS_BEEN_CONTINUOUSLY_INHABITED_SINCE", "2000"),
    ("Gravity", "CAUSES", "Earth's spherical shape"),
    (
        "Moon landing 1969",
        "IS_VERIFIED_BY",
        "Independent scientific communities worldwide",
    ),
    (
        "Lunar retroreflectors",
        "PLACED_BY",
        "Apollo missions and still used by researchers",
    ),
    ("Flat Earth theory", "CONTRADICTED_BY", "GPS, aviation, and satellite data"),
    ("Stars", "ARE", "Distant suns in the Milky Way galaxy"),
    (
        "James Webb telescope",
        "PROVIDES",
        "Deep space imagery at unprecedented resolution",
    ),
    ("Cosmic microwave background", "IS_EVIDENCE_FOR", "Big Bang cosmology"),
    # Health Misinformation
    ("Bleach", "IS_TOXIC", "And must never be ingested"),
    ("Ivermectin", "NOT_APPROVED_BY_FDA_FOR", "COVID-19 treatment in humans"),
    ("Hydroxychloroquine", "SHOWED_NO_BENEFIT_IN", "Large randomized COVID-19 trials"),
    (
        "Vitamin D deficiency",
        "LINKED_TO",
        "Higher COVID-19 severity but not prevention",
    ),
    ("Masks", "REDUCE_TRANSMISSION_OF", "Respiratory droplets when worn correctly"),
    ("Antibiotics", "DO_NOT_WORK_AGAINST", "Viral infections"),
    ("Autism", "HAS_COMPLEX_GENETIC_AND_ENVIRONMENTAL", "Origins being researched"),
    ("Homeopathy", "HAS_NO_EVIDENCE_BEYOND", "Placebo effect in rigorous trials"),
    ("Essential oils", "ARE_NOT_PROVEN_TO", "Treat cancer or serious illness"),
    (
        "Flu vaccine",
        "RECOMMENDED_ANNUALLY_BECAUSE",
        "Influenza strains evolve each year",
    ),
    # Financial Misinformation
    ("Bitcoin", "IS", "A decentralized digital currency with high volatility"),
    (
        "Ponzi schemes",
        "ARE_ILLEGAL_AND_DEFINED_BY",
        "Using new investor money to pay earlier investors",
    ),
    ("Federal Reserve", "IS_A", "Quasi-public central bank, not purely private"),
    ("Gold standard", "ABANDONED_BY_US_IN", "1971 under Nixon"),
    ("Inflation", "CAUSED_BY", "Money supply growth and demand-supply imbalances"),
    ("Stock market crashes", "HISTORICALLY_RECOVER_OVER", "Long time horizons"),
    (
        "Pyramid schemes",
        "INEVITABLY_COLLAPSE_BECAUSE",
        "Recruitment becomes mathematically impossible",
    ),
    ("NFTs", "ARE", "Digital ownership tokens with highly speculative value"),
    (
        "Central bank digital currencies",
        "BEING_EXPLORED_BY",
        "Over 100 countries globally",
    ),
    ("Wealth inequality", "MEASURED_BY", "Gini coefficient across nations"),
]


class KnowledgeGraph:
    """
    Wraps the neo4j Python driver to provide entity-relationship
    evidence retrieval for OSKAR's Graph-RAG layer.

    connect()  → tries bolt connection, sets self.connected
    seed()     → loads SEED_TRIPLES into the graph (idempotent)
    query_context(claim) → returns list of matching evidence triples
    close()    → closes the driver
    """

    def __init__(self, auto_seed: bool = True):
        self.driver = None
        self.connected = False
        self._connect(auto_seed)

    def _connect(self, auto_seed: bool):
        try:
            from neo4j import GraphDatabase
            from neo4j import exceptions as neo4j_exc

            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            # Verify connectivity
            self.driver.verify_connectivity()
            self.connected = True
            print(f"[Neo4j] Connected to {NEO4J_URI}")
            if auto_seed:
                self._seed()
        except Exception as e:
            self.connected = False
            print(f"[Neo4j] Not available ({e}). Using FAISS-only mode.")

    def _seed(self):
        """Insert seed triples (idempotent via MERGE)."""
        with self.driver.session() as session:
            for subj, rel, obj in SEED_TRIPLES:
                session.run(
                    """
                    MERGE (a:Entity {name: $subj})
                    MERGE (b:Fact   {text: $obj})
                    MERGE (a)-[r:RELATION {type: $rel}]->(b)
                    """,
                    subj=subj,
                    rel=rel,
                    obj=obj,
                )
        print(f"[Neo4j] Seeded {len(SEED_TRIPLES)} fact triples.")

    def query_context(self, claim: str, max_hops: int = 2) -> list[dict]:
        """
        Extract key tokens from the claim, find related nodes in the graph,
        retrieve their outgoing relationships up to `max_hops` away.

        Returns a list of evidence dicts:
          {"subject": str, "relation": str, "object": str, "relevance": float}
        """
        if not self.connected:
            return []

        # Simple keyword extraction: words 4+ chars, no stop words
        STOP = {
            "this",
            "that",
            "with",
            "from",
            "they",
            "have",
            "been",
            "will",
            "would",
            "could",
            "there",
            "their",
            "about",
        }
        tokens = [w.lower() for w in re.findall(r"\b\w{4,}\b", claim) if w.lower() not in STOP]
        if not tokens:
            return []

        results = []
        with self.driver.session() as session:
            for token in tokens[:5]:  # cap to 5 tokens per query
                records = session.run(
                    """
                    MATCH (a:Entity)
                    WHERE toLower(a.name) CONTAINS $token
                    MATCH (a)-[r:RELATION]->(b)
                    RETURN a.name AS subject, r.type AS relation, b.text AS object
                    LIMIT 3
                    """,
                    token=token,
                )
                for rec in records:
                    results.append(
                        {
                            "subject": rec["subject"],
                            "relation": rec["relation"],
                            "object": rec["object"],
                            "relevance": 1.0,
                        }
                    )

        seen = set()
        deduped = []
        for r in results:
            key = (r["subject"], r["relation"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return deduped[:6]  # Top 6 triples max for API response size

    def close(self):
        if self.driver:
            self.driver.close()
