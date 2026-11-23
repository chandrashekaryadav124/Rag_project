AI COMMERCE — CUSTOMER RAG FINAL KNOWLEDGE BASE
Version: consolidated customer-only pack

CONTENTS
1. 01_Customer_Knowledge_Base_Data_and_RAG_Guidelines.pdf
   Consolidates the source/usage guide, 550-product catalog reference,
   customer review insights, support-ticket reference, and aggregate
   orders/delivery/returns reference.

2. 02_Customer_Policies_and_FAQ_PROPOSED_NOT_APPROVED.pdf
   Consolidates proposed return/refund, cancellation, shipping/delivery,
   payment, and account/privacy FAQ drafts.

IMPORTANT ACCURACY AND RAG RULES
- This pack is for the CUSTOMER role only. Seller, Admin, Support, and
  Operations knowledge bases are intentionally not included.
- Dataset figures are a snapshot of the uploaded project dataset, not live
  platform data. Never answer a customer's current stock, price, order,
  payment, shipment, return, or refund status from this snapshot.
- Use authenticated backend/API queries to the live application database
  for personalized or changing records. Enforce customer ownership checks
  server-side; never rely on prompt instructions as authorization.
- Use approved public customer documents for policy answers. The policy PDF
  is explicitly PROPOSED and NOT APPROVED. Do not present it as current
  company policy or promise its draft timeframes.
- The 10-day return period is a proposed 10-CALENDAR-DAY interpretation.
  The event that starts the clock is not approved. Do not calculate a
  deadline until the business confirms the trigger.
- The 7–10 business day refund estimate is proposed only after refund
  approval and initiation; it is not an approved guarantee and bank posting
  may differ.
- Delivery supports both courier-partner and seller-managed models as a
  project planning choice; responsibilities, coverage and service promises
  still require business approval.
- Derived product_knowledge, review_text and support_tickets content is
  semi-synthetic. Do not describe generated review text as verbatim
  customer-written reviews or generated ticket examples as actual agent
  conversations or approved SOP.
- Do not ingest raw SQL dumps, passwords, password hashes, personal
  identifiers, private customer records, payment secrets, or unrestricted
  transaction exports into a shared customer Chroma collection.
- Never expose internal cost_price or return_rate_baseline fields to customers.
- When evidence is missing or conflicting, say that the information is not
  confirmed and route the customer to the in-app support path.

SUGGESTED METADATA
role=customer
document_type=product_reference | review_insight | support_reference |
              transaction_dataset_reference | policy_or_faq
policy_status=proposed (for PDF 02 only)
policy_version=0.1 (for PDF 02 only)
evidence_type=verified_source_field | derived_aggregate |
              semi_synthetic_example | proposed_wording
effective_date=null until approved

RECOMMENDED CHROMA INGESTION
- Ingest PDF 01 as reference-only content with clear evidence labels.
- Keep PDF 02 out of the production policy collection until approved.
  It may be stored in a separate development collection with
  policy_status=proposed, and the retriever must not treat it as authoritative.
- Keep live customer/order/payment/shipment/return data out of shared Chroma.
  Fetch those records from the authenticated backend at question time.
